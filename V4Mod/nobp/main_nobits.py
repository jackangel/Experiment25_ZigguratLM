import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
import tiktoken
import os
import time
import math
import random
import argparse
from collections import Counter
import glob

from torch.utils.data import IterableDataset, DataLoader, TensorDataset
from torch.cuda.amp import GradScaler
import pyarrow.parquet as pq

try:
    import pandas as pd
except ImportError:
    pd = None

from model import ModelConfig, DiscretizedManifoldTransformer

class GlobalPerturbationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, layer_id):
        # Standard linear forward pass
        output = F.linear(x, weight, bias)
        
        # Save for backward
        ctx.save_for_backward(x, weight, bias)
        ctx.layer_id = layer_id
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias = ctx.saved_tensors
        layer_id = ctx.layer_id
        
        # Ensure all tensors have the same dtype (use weight's dtype as reference)
        target_dtype = weight.dtype
        grad_output = grad_output.to(target_dtype)
        x = x.to(target_dtype)
        
        # Standard gradients
        grad_input = grad_weight = grad_bias = None
        
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.matmul(weight)
        
        if ctx.needs_input_grad[1]:
            # Reshape for proper matrix multiplication
            if grad_output.dim() > 2:
                # Flatten all dimensions except the last one
                grad_output_flat = grad_output.view(-1, grad_output.size(-1))
                x_flat = x.view(-1, x.size(-1))
                grad_weight = grad_output_flat.transpose(-2, -1).matmul(x_flat)
            else:
                grad_weight = grad_output.transpose(-2, -1).matmul(x)
            
        if bias is not None and ctx.needs_input_grad[2]:
            # Sum over all dimensions except the last one
            grad_bias = grad_output
            while grad_bias.dim() > 1:
                grad_bias = grad_bias.sum(0)
        
        return grad_input, grad_weight, grad_bias, None

class EnhancedGPLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, layer_id=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.layer_id = layer_id
        
        # Standard linear layer parameters
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features)) if bias else None
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        """Forward pass using Global Perturbation."""
        return GlobalPerturbationFunction.apply(x, self.weight, self.bias, self.layer_id)

class HybridDiscretizedManifoldTransformer(DiscretizedManifoldTransformer):
    """DMT with Global Perturbation integration for select layers."""
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        # Convert specific layers to GP (attention projections and some MLPs)
        self._convert_to_gp_layers()
        
    def _convert_to_gp_layers(self):
        """Convert strategic layers to Global Perturbation."""
        layer_id = 0
        
        for stage_idx, stage in enumerate(self.stages):
            for block_idx, block in enumerate(stage):
                # Check attention module (it's called 'attention', not 'attn')
                if hasattr(block, 'attention'):
                    attention_module = block.attention
                    # Find all Linear layers in attention
                    for attr_name in dir(attention_module):
                        if not attr_name.startswith('_'):
                            attr = getattr(attention_module, attr_name)
                            if isinstance(attr, nn.Linear):
                                new_layer = EnhancedGPLinear(
                                    attr.in_features, 
                                    attr.out_features, 
                                    bias=attr.bias is not None,
                                    layer_id=layer_id
                                )
                                with torch.no_grad():
                                    new_layer.weight.data.copy_(attr.weight.data)
                                    if new_layer.bias is not None:
                                        new_layer.bias.data.copy_(attr.bias.data)
                                
                                setattr(attention_module, attr_name, new_layer)
                                layer_id += 1
                                print(f"Converted attention.{attr_name} in stage {stage_idx}, block {block_idx}")
                
                # Check MLP module (it's a ModuleList)
                if hasattr(block, 'mlp') and isinstance(block.mlp, nn.ModuleList):
                    # Convert first layer of MLP only
                    if len(block.mlp) > 0 and isinstance(block.mlp[0], nn.Linear):
                        old_layer = block.mlp[0]
                        new_layer = EnhancedGPLinear(
                            old_layer.in_features, 
                            old_layer.out_features, 
                            bias=old_layer.bias is not None,
                            layer_id=layer_id
                        )
                        with torch.no_grad():
                            new_layer.weight.data.copy_(old_layer.weight.data)
                            if new_layer.bias is not None:
                                new_layer.bias.data.copy_(old_layer.bias.data)
                        
                        block.mlp[0] = new_layer
                        layer_id += 1
                        print(f"Converted mlp[0] in stage {stage_idx}, block {block_idx}")
        
        print(f"Successfully converted {layer_id} layers to Global Perturbation")

class TripleHybridOptimizer:
    """Handles GP, Standard, and VQ parameters with different learning rates."""
    
    def __init__(self, model, learning_rate=3e-4, vq_lr_multiplier=0.1, gp_lr_multiplier=0.1, weight_decay=1e-4):
        self.model = model
        self.learning_rate = learning_rate
        self.vq_learning_rate = learning_rate * vq_lr_multiplier
        self.gp_learning_rate = learning_rate * gp_lr_multiplier
        self.weight_decay = weight_decay
        
        # Categorize parameters
        self.gp_params = []
        self.standard_params = []
        self.vq_params = []
        
        for name, param in model.named_parameters():
            if 'vq.quantizers' in name:
                self.vq_params.append(param)
            else:
                # Check if this parameter belongs to a GP layer by finding the module
                is_gp_param = False
                try:
                    # Navigate to the parent module
                    module_path = name.split('.')[:-1]  # Remove parameter name
                    current_module = model
                    for part in module_path:
                        current_module = getattr(current_module, part)
                    
                    # Check if parent module is EnhancedGPLinear
                    if isinstance(current_module, EnhancedGPLinear):
                        self.gp_params.append((name, param))
                        is_gp_param = True
                except:
                    pass
                
                if not is_gp_param:
                    self.standard_params.append(param)
        
        # Create optimizers for non-GP parameters
        self.optimizer_standard = torch.optim.AdamW(
            self.standard_params, 
            lr=learning_rate, 
            betas=(0.9, 0.95),
            weight_decay=weight_decay
        ) if self.standard_params else None
        
        self.optimizer_vq = torch.optim.AdamW(
            self.vq_params, 
            lr=self.vq_learning_rate, 
            betas=(0.9, 0.95),
            weight_decay=weight_decay
        ) if self.vq_params else None
        
        print(f"TripleHybridOptimizer initialized:")
        print(f"  GP parameters: {len(self.gp_params)} ({sum(p.numel() for _, p in self.gp_params):,} total)")
        print(f"  Standard parameters: {len(self.standard_params)} ({sum(p.numel() for p in self.standard_params):,} total)")
        print(f"  VQ parameters: {len(self.vq_params)} ({sum(p.numel() for p in self.vq_params):,} total)")
    
    def zero_grad(self):
        if self.optimizer_standard:
            self.optimizer_standard.zero_grad(set_to_none=True)
        if self.optimizer_vq:
            self.optimizer_vq.zero_grad(set_to_none=True)
        # GP parameters don't use .grad, so no need to zero them
    
    def step(self):
        # Step standard and VQ optimizers
        if self.optimizer_standard:
            self.optimizer_standard.step()
        if self.optimizer_vq:
            self.optimizer_vq.step()
        
        # Manual weight decay for GP parameters
        if self.weight_decay > 0:
            with torch.no_grad():
                for name, param in self.gp_params:
                    param.data.mul_(1 - self.gp_learning_rate * self.weight_decay)
    
    def state_dict(self):
        state = {}
        if self.optimizer_standard:
            state['optimizer_standard'] = self.optimizer_standard.state_dict()
        if self.optimizer_vq:
            state['optimizer_vq'] = self.optimizer_vq.state_dict()
        return state
    
    def load_state_dict(self, state_dict):
        if 'optimizer_standard' in state_dict and self.optimizer_standard:
            self.optimizer_standard.load_state_dict(state_dict['optimizer_standard'])
        if 'optimizer_vq' in state_dict and self.optimizer_vq:
            self.optimizer_vq.load_state_dict(state_dict['optimizer_vq'])

def hybrid_dmt_training_step(model, batch, scaler, grad_accum_steps, device, config):
    """Training step that handles DMT complexity + GP training."""
    
    xb = batch[:, :config.block_size].to(device, non_blocking=True)
    yb = batch[:, 1:config.block_size+1].to(device, non_blocking=True)
    
    with torch.amp.autocast(device_type=device):
        try:
            # Get all the complex losses from DMT
            logits, (ce_loss, q_loss, e_loss, conf_loss) = model(xb, yb)
            
            # Calculate composite losses
            main_loss = (ce_loss + (conf_loss * config.confidence_loss_weight if config.use_confidence_circuit else 0)) / grad_accum_steps
            vq_optimizer_loss = q_loss / grad_accum_steps
            
            # Check for numerical issues
            if not (torch.isfinite(main_loss) and torch.isfinite(vq_optimizer_loss)):
                print(f"Non-finite losses detected, skipping step")
                return 0.0, 0.0, 0.0, 0.0
                
        except Exception as e:
            print(f"Error in forward pass: {e}")
            return 0.0, 0.0, 0.0, 0.0
    
    try:
        # Backward pass with retain_graph for multiple losses
        scaler.scale(main_loss).backward(retain_graph=True)
        scaler.scale(vq_optimizer_loss).backward()
        
        return (main_loss.item() * grad_accum_steps, 
                ce_loss.item(), 
                conf_loss.item(), 
                q_loss.item())
    except Exception as e:
        print(f"Error in backward pass: {e}")
        return 0.0, 0.0, 0.0, 0.0

@torch.no_grad()
def display_bucket_contents(model, tokenizer, device, analysis_tokens, stage_to_show=0, layer_to_show=0, top_k=8):
    model.eval()
    print(f"\n--- BUCKET ANALYSIS (Stage {stage_to_show}, Layer {layer_to_show}) ---")
    tokens = analysis_tokens.unsqueeze(0).to(device)
    
    try:
        _, all_stage_indices = model(tokens, return_indices=True)
        config = model.config
        seq_len_multiplier = (0.5 ** stage_to_show)
        
        if stage_to_show >= len(all_stage_indices):
            print(f"Error: Stage {stage_to_show} not found.")
            model.train()
            return
            
        layer_indices = all_stage_indices[stage_to_show][layer_to_show]
        
        if stage_to_show > 0:
            print("--- Analyzing Multi-Token Concepts ---")
            for level_idx in range(config.vq_levels): 
                print(f"\n--- VQ HIERARCHY LEVEL {level_idx} ---\n  Top 5 most used buckets: {Counter(layer_indices[level_idx].flatten().tolist()).most_common(5)}")
        else:
            bucket_contents = [[Counter() for _ in range(config.num_codes)] for _ in range(config.vq_levels)]
            expected_seq_len = min(int(config.block_size * seq_len_multiplier), tokens.shape[1])
            
            for level_idx in range(config.vq_levels):
                for seq_idx in range(expected_seq_len):
                    token_id = tokens[0, seq_idx].item()
                    bucket_id = layer_indices[level_idx, 0, seq_idx].item()
                    try: 
                        bucket_contents[level_idx][bucket_id][tokenizer.decode([token_id])] += 1
                    except: 
                        continue
            
            for level_idx in range(config.vq_levels):
                print(f"\n--- VQ HIERARCHY LEVEL {level_idx} ---")
                used_buckets = sorted([(i, c) for i, c in enumerate(bucket_contents[level_idx]) if c], key=lambda item: len(item[1]), reverse=True)
                for bucket_id, counter in used_buckets[:5]: 
                    print(f"  Bucket [{bucket_id:4d}] (Unique: {len(counter):,d}): {counter.most_common(top_k)}")
    except Exception as e:
        print(f"Error in bucket analysis: {e}")
    
    print("--- End of Bucket Analysis ---\n")
    model.train()

@torch.no_grad()
def generate(model, tokenizer, device, prompt, max_new_tokens=50):
    model.eval()
    prompt_tokens = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    
    try:
        for _ in range(max_new_tokens):
            idx_cond = prompt_tokens[:, -model.config.block_size:]
            logits, _ = model(idx_cond)
            logits, probs = logits[:, -1, :], F.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            prompt_tokens = torch.cat((prompt_tokens, next_token), dim=1)
        
        decoded = tokenizer.decode(prompt_tokens.squeeze().tolist())
        print(f"--- PROMPT: '{prompt}' ---\nGENERATION: {decoded}\n-------------------------------------")
    except Exception as e:
        print(f"Error in generation: {e}")
    
    model.train()

def chat(checkpoint_path, device, max_new_tokens=256):
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    tokenizer = tiktoken.get_encoding("cl100k_base")
    config.vocab_size = tokenizer.n_vocab
    model = HybridDiscretizedManifoldTransformer(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"\nHybrid DMT+GP Model loaded ({sum(p.numel() for p in model.parameters())/1e6:.2f}M params). Chat away! (type 'exit' to quit)")
    
    while True:
        prompt = input("You: ")
        if prompt.lower() in ['exit', 'quit']: 
            break
        
        try:
            prompt_tokens = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
            print("Model:", end=" ", flush=True)
            
            for _ in range(max_new_tokens):
                idx_cond = prompt_tokens[:, -model.config.block_size:]
                with torch.no_grad(): 
                    logits, _ = model(idx_cond)
                logits, probs = logits[:, -1, :], F.softmax(logits[:, -1, :], dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                if hasattr(tokenizer, 'eot_token') and next_token.item() == tokenizer.eot_token: 
                    break
                prompt_tokens = torch.cat((prompt_tokens, next_token), dim=1)
                print(tokenizer.decode([next_token.item()]), end="", flush=True)
            print("\n")
        except Exception as e:
            print(f"Error in chat: {e}\n")

class TokenStreamDataset(IterableDataset):
    def __init__(self, file_paths, tokenizer, text_column, block_size):
        self.file_paths = file_paths
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.block_size = block_size
        self.buffer = []
    
    def __iter__(self):
        random.shuffle(self.file_paths)
        for file_path in self.file_paths:
            print(f"\nStreaming from {os.path.basename(file_path)}...")
            try:
                parquet_file = pq.ParquetFile(file_path)
                for batch in parquet_file.iter_batches(columns=[self.text_column]):
                    df_chunk = batch.to_pandas()
                    for text in df_chunk[self.text_column].astype(str):
                        tokens = self.tokenizer.encode(text)
                        self.buffer.extend(tokens)
                        while len(self.buffer) >= self.block_size + 1:
                            chunk = self.buffer[:self.block_size + 1]
                            self.buffer = self.buffer[self.block_size:]
                            yield torch.tensor(chunk, dtype=torch.long)
            except Exception as e:
                print(f"  [Error] Failed to stream from {os.path.basename(file_path)}: {e}. Skipping.")
                continue

if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    parser = argparse.ArgumentParser(description="Train or chat with a Hybrid DMT+GP model.")
    parser.add_argument('--parquet_folder', type=str, default=None, help='Path to a folder of parquet files for training.')
    parser.add_argument('--text_column', type=str, default='text', help='The name of the text column in the parquet files.')
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    BATCH_SIZE = 6
    GRAD_ACCUM_STEPS = 8
    LEARNING_RATE = 3e-4
    EVAL_EVERY = 20000
    CHECKPOINT_DIR = "checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    checkpoint_files = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "*.pth")))
    latest_checkpoint = checkpoint_files[-1] if checkpoint_files else None
    resume_from_checkpoint = None
    
    if latest_checkpoint:
        print(f"Found latest checkpoint: {os.path.basename(latest_checkpoint)}")
        while True:
            choice = input("Choose an action: [C]ontinue training, start [Chat] mode, or start [N]ew training? (C/Chat/N): ").lower()
            if choice in ['c', 'continue']: 
                resume_from_checkpoint = latest_checkpoint
                break
            elif choice in ['chat']: 
                chat(latest_checkpoint, device)
                exit()
            elif choice in ['n', 'new']: 
                break
            else: 
                print("Invalid choice.")
    
    tokenizer = tiktoken.get_encoding("p50k_base")
    global_step = 0
    
    if resume_from_checkpoint:
        checkpoint = torch.load(resume_from_checkpoint, map_location=device)
        config = checkpoint['config']
        model = HybridDiscretizedManifoldTransformer(config).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        global_step = checkpoint.get('global_step', 0)
        print(f"Resuming from global step {global_step}.")
    else:
        config = ModelConfig()
        config.vocab_size = tokenizer.n_vocab
        model = HybridDiscretizedManifoldTransformer(config).to(device)

    # Initialize the triple hybrid optimizer
    hybrid_optimizer = TripleHybridOptimizer(
        model, 
        learning_rate=LEARNING_RATE,
        vq_lr_multiplier=0.1,  # VQ gets 10% of main LR
        gp_lr_multiplier=0.05  # GP gets 5% of main LR (more conservative)
    )
    
    scaler = torch.amp.GradScaler('cuda')  # Fixed deprecation warning
    
    if resume_from_checkpoint and 'optimizer_state_dict' in checkpoint:
        hybrid_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scaler_state_dict' in checkpoint: 
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    print(f"Hybrid DMT+GP Model initialized with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters.")
    print(f"Effective batch size: {BATCH_SIZE * GRAD_ACCUM_STEPS}")
    
    if args.parquet_folder:
        if pd is None: 
            raise ImportError("Pandas is not installed. Please run 'pip install pandas pyarrow'.")
        
        parquet_files = sorted(glob.glob(os.path.join(args.parquet_folder, '*.parquet')))
        if not parquet_files: 
            raise ValueError(f"No .parquet files found in {args.parquet_folder}")
        
        print(f"Found {len(parquet_files)} parquet files for training.")
        print(f"Starting Enhanced Hybrid DMT+GP Training...")
        
        dataset = TokenStreamDataset(parquet_files, tokenizer, args.text_column, config.block_size)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0, pin_memory=True)
        
        model.train()
        
        for micro_step, batch_chunk in enumerate(dataloader):
            # Use the enhanced training step
            loss, ce_loss, conf_loss, q_loss = hybrid_dmt_training_step(
                model, batch_chunk, scaler, GRAD_ACCUM_STEPS, device, config
            )
            
            # Only proceed with optimization if forward pass succeeded
            if loss > 0:  # Forward pass succeeded
                if (micro_step + 1) % GRAD_ACCUM_STEPS == 0:
                    try:
                        # Enhanced gradient clipping for different parameter types
                        if hybrid_optimizer.optimizer_standard:
                            scaler.unscale_(hybrid_optimizer.optimizer_standard)
                        if hybrid_optimizer.optimizer_vq:
                            scaler.unscale_(hybrid_optimizer.optimizer_vq)
                        
                        # Separate clipping for different parameter types
                        gp_params = [p for _, p in hybrid_optimizer.gp_params]
                        if gp_params:
                            torch.nn.utils.clip_grad_norm_(gp_params, 0.5)  # Conservative for GP
                        if hybrid_optimizer.standard_params:
                            torch.nn.utils.clip_grad_norm_(hybrid_optimizer.standard_params, 1.0)
                        if hybrid_optimizer.vq_params:
                            torch.nn.utils.clip_grad_norm_(hybrid_optimizer.vq_params, 1.0)
                        
                        # Step all optimizers
                        hybrid_optimizer.step()
                        scaler.update()
                        hybrid_optimizer.zero_grad()
                        
                        global_step += 1
                        
                        if global_step % 100 == 0:
                            print(f"Hybrid DMT+GP Step {global_step} | Loss {loss:.4f} (CE: {ce_loss:.4f}, Conf: {conf_loss:.4f}, Q: {q_loss:.4f})")
                        
                        if global_step > 0 and global_step % EVAL_EVERY == 0:
                            generate(model, tokenizer, device, prompt="The meaning of life is")
                            analysis_tokens_str = "This is a sample text for analyzing the model's internal state."
                            analysis_sample = torch.tensor(tokenizer.encode(analysis_tokens_str), dtype=torch.long)[:config.block_size]
                            display_bucket_contents(model, tokenizer, device, analysis_sample, stage_to_show=0, layer_to_show=0)
                            
                            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"hybrid_dmt_gp_step_{global_step}.pth")
                            print(f"Saving checkpoint to {checkpoint_path}")
                            
                            torch.save({
                                'global_step': global_step, 
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': hybrid_optimizer.state_dict(),
                                'scaler_state_dict': scaler.state_dict(), 
                                'config': config
                            }, checkpoint_path)
                    
                    except Exception as e:
                        print(f"Error in optimization step: {e}")
                        hybrid_optimizer.zero_grad()
            else:
                # Reset gradients on failed forward pass
                hybrid_optimizer.zero_grad()
    
    print("\n--- Hybrid DMT+GP Training Finished ---")
