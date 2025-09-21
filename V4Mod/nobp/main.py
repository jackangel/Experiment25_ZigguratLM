# main.py
# Final, fully corrected and refactored version.

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
from typing import Optional, Dict, Any, List

from torch.utils.data import IterableDataset, DataLoader
import pyarrow.parquet as pq

try:
    import pandas as pd
except ImportError:
    pd = None

from model import ModelConfig, DiscretizedManifoldTransformer, FastExponentialGELU, ResidualVectorQuantizer

# --- JIT-COMPILED UPDATE FUNCTIONS ---
@torch.jit.script
def jit_linear_update(weight: torch.Tensor, bias: Optional[torch.Tensor], target: torch.Tensor, layer_input: torch.Tensor, lr: float) -> torch.Tensor:
    if layer_input.dim() > 2: layer_input = layer_input.reshape(-1, layer_input.size(-1))
    if target.dim() > 2: target = target.reshape(-1, target.size(-1))
    weight_update = torch.matmul(target.T, layer_input)
    weight.add_(weight_update, alpha=lr)
    if bias is not None:
        bias_update = target.sum(dim=0)
        bias.add_(bias_update, alpha=lr)
    parent_target = torch.matmul(target, weight.data)
    return parent_target

@torch.jit.script
def jit_layernorm_update(ln_weight: Optional[torch.Tensor], ln_bias: Optional[torch.Tensor], target: torch.Tensor, layer_input: torch.Tensor, mean: torch.Tensor, var: torch.Tensor, eps: float, lr: float) -> torch.Tensor:
    normalized_input = (layer_input - mean) / torch.sqrt(var + eps)
    if ln_weight is not None:
        weight_update = (target * normalized_input).sum(dim=list(range(target.dim()-1)))
        ln_weight.add_(weight_update, alpha=lr)
        target = target * ln_weight
    if ln_bias is not None:
        bias_update = target.sum(dim=list(range(target.dim()-1)))
        ln_bias.add_(bias_update, alpha=lr)
    parent_target = target / torch.sqrt(var + eps)
    return parent_target

@torch.jit.script
def jit_activation_inverse(target: torch.Tensor, pre_activation: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    exp_term = torch.exp(-0.5 * pre_activation * pre_activation)
    derivative = 1.0 + alpha * exp_term - alpha * pre_activation * pre_activation * exp_term
    return target * derivative

# --- REWRITTEN ARCHITECTURE-AWARE NO-BACKPROPAGATION OPTIMIZER ---

class NoBackpropOptimizer:
    def __init__(self, model: nn.Module, learning_rate: float = 1e-5):
        self.model = model
        self.learning_rate = learning_rate
        self.inputs_cache: Dict[nn.Module, torch.Tensor] = {}
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []

    def _cache_input_hook(self, module: nn.Module, inputs: tuple, output: Any):
        if isinstance(inputs, tuple) and len(inputs) > 0 and isinstance(inputs[0], torch.Tensor):
            self.inputs_cache[module] = inputs[0].detach()

    def _register_hooks(self):
        self.inputs_cache.clear() # Ensure cache is clean before a new forward pass
        for module in self.model.modules():
            self.hooks.append(module.register_forward_hook(self._cache_input_hook))

    def _remove_hooks(self):
        for handle in self.hooks: handle.remove()
        self.hooks.clear()

    def _update_and_propagate(self, module: nn.Module, target: torch.Tensor) -> torch.Tensor:
        module_input = self.inputs_cache.get(module)
        if module_input is None: return target

        if isinstance(module, nn.Linear):
            parent_target = jit_linear_update(module.weight, module.bias, target, module_input, self.learning_rate)
            return parent_target.view(module_input.shape)
        elif isinstance(module, nn.LayerNorm):
            mean = module_input.mean(dim=-1, keepdim=True)
            var = module_input.var(dim=-1, keepdim=True, unbiased=False)
            parent_target = jit_layernorm_update(module.weight, module.bias, target, module_input, mean, var, module.eps, self.learning_rate)
            return parent_target.view(module_input.shape)
        elif isinstance(module, FastExponentialGELU):
            return jit_activation_inverse(target, module_input, module.alpha)
        elif isinstance(module, nn.ReLU):
            return target * (module_input > 0).float()
        
        return target

    def step(self, xb: torch.Tensor, yb: torch.Tensor) -> Dict[str, float]:
        if xb.numel() == 0: return {}

        with torch.no_grad():
            self._register_hooks()
            model_output = self.model(xb, yb)
            self._remove_hooks()
            
            losses, logits = model_output['losses'], model_output['logits']
            self._vq_local_update(losses['e_loss'])
            final_hidden_state = model_output['final_hidden_state']
            
            y_true_one_hot = F.one_hot(yb, num_classes=self.model.config.vocab_size).float()
            lm_target = y_true_one_hot - F.softmax(logits, dim=-1)
            conf_target = 0.0
            if self.model.config.use_confidence_circuit:
                conf_pred = self.model.confidence_head(final_hidden_state)
                correctness_targets = (logits.argmax(dim=-1) == yb).float()
                conf_target = (correctness_targets.unsqueeze(-1) - torch.sigmoid(conf_pred))
            
            parent_target_lm = self._update_and_propagate(self.model.lm_head, lm_target)
            parent_target_conf = 0.0
            if self.model.config.use_confidence_circuit:
                current_conf_target = self._update_and_propagate(self.model.confidence_head[2], conf_target)
                current_conf_target = self._update_and_propagate(self.model.confidence_head[1], current_conf_target)
                parent_target_conf = self._update_and_propagate(self.model.confidence_head[0], current_conf_target)
            
            current_target = parent_target_lm + parent_target_conf
            current_target = self._update_and_propagate(self.model.ln_f, current_target)

            for block in reversed(self.model.stages[0]):
                block_output_target = current_target
                
                # Path back through MLP branch
                mlp_branch_target = self._update_and_propagate(block.mlp[2], block_output_target)
                mlp_branch_target = self._update_and_propagate(block.mlp[1], mlp_branch_target)
                mlp_branch_target = self._update_and_propagate(block.mlp[0], mlp_branch_target)
                mlp_branch_target = self._update_and_propagate(block.ln_2, mlp_branch_target)
                
                # Path back through Attention branch
                attn_branch_target = self._update_and_propagate(block.attention.ln, block_output_target)
                attn_branch_target = self._update_and_propagate(block.attention.proj, attn_branch_target)
                attn_branch_target = self._update_and_propagate(block.ln_1, attn_branch_target)

                # Sum targets for the input of the residual block
                current_target = attn_branch_target + mlp_branch_target
            
            # -- CORRECTED EMBEDDING & CONTROLLER REVERSE PASS --
            base_emb = self.inputs_cache[self.model.controller]
            abstraction_logits = self.model.controller(base_emb)
            abstraction_weights = F.softmax(abstraction_logits, dim=-1)
            weights = abstraction_weights.permute(2, 0, 1).unsqueeze(-1)
            
            rolled_embeddings = [base_emb]
            for op in self.model.transformation_ops:
                rolled_embeddings.append(op(self.inputs_cache[op]))
            stacked_embeddings = torch.stack(rolled_embeddings, dim=0)

            target_for_weights_raw = current_target.unsqueeze(0) * stacked_embeddings
            target_for_abstraction_weights = target_for_weights_raw.sum(dim=-1).permute(1, 2, 0)
            
            controller_target = target_for_abstraction_weights
            controller_target = self._update_and_propagate(self.model.controller[2], controller_target)
            controller_target = self._update_and_propagate(self.model.controller[1], controller_target)
            base_emb_target_from_controller = self._update_and_propagate(self.model.controller[0], controller_target)

            target_for_stacked = current_target.unsqueeze(0) * weights
            base_emb_target_from_embeddings = target_for_stacked[0]
            for i, op in enumerate(self.model.transformation_ops):
                base_emb_target_from_embeddings += self._update_and_propagate(op, target_for_stacked[i+1])
            
            total_base_emb_target = base_emb_target_from_controller + base_emb_target_from_embeddings
            
            wte_input_indices = xb.flatten()
            wte_target = total_base_emb_target.reshape(-1, total_base_emb_target.size(-1))
            self.model.wte.weight.data.index_add_(0, wte_input_indices, wte_target, alpha=self.learning_rate)
            
        return {name: loss.item() for name, loss in losses.items() if loss is not None and torch.is_tensor(loss)}

    def _vq_local_update(self, e_loss: torch.Tensor):
        if not torch.is_tensor(e_loss): return
        for module in self.model.modules():
            if isinstance(module, ResidualVectorQuantizer):
                for quantizer in module.quantizers:
                    update_direction = torch.randn_like(quantizer.embedding.weight.data) * e_loss
                    quantizer.embedding.weight.data.add_(update_direction, alpha=-self.learning_rate * 10)

# --- UTILITY FUNCTIONS & MAIN SCRIPT ---
# (The rest of the file remains unchanged)

@torch.no_grad()
def display_bucket_contents(model, tokenizer, device, analysis_tokens, stage_to_show=0, layer_to_show=0, top_k=8):
    model.eval()
    print(f"\n--- BUCKET ANALYSIS (Stage {stage_to_show}, Layer {layer_to_show}) ---")
    tokens = analysis_tokens.unsqueeze(0).to(device)
    output = model(tokens)
    all_stage_indices = output['all_indices']
    config = model.config
    seq_len_multiplier = (0.5 ** stage_to_show)
    if stage_to_show >= len(all_stage_indices):
        print(f"Error: Stage {stage_to_show} not found.")
        model.train()
        return
    layer_indices = all_stage_indices[stage_to_show][layer_to_show]
    if stage_to_show > 0:
        print("--- Analyzing Multi-Token Concepts ---")
        for level_idx in range(config.vq_levels): print(f"\n--- VQ HIERARCHY LEVEL {level_idx} ---\n  Top 5 most used buckets: {Counter(layer_indices[level_idx].flatten().tolist()).most_common(5)}")
    else:
        bucket_contents = [[Counter() for _ in range(config.num_codes)] for _ in range(config.vq_levels)]
        expected_seq_len = min(int(config.block_size * seq_len_multiplier), tokens.shape[1])
        for level_idx in range(config.vq_levels):
            for seq_idx in range(expected_seq_len):
                token_id = tokens[0, seq_idx].item(); bucket_id = layer_indices[level_idx, 0, seq_idx].item()
                try: bucket_contents[level_idx][bucket_id][tokenizer.decode([token_id])] += 1
                except: continue
        for level_idx in range(config.vq_levels):
            print(f"\n--- VQ HIERARCHY LEVEL {level_idx} ---")
            used_buckets = sorted([(i, c) for i, c in enumerate(bucket_contents[level_idx]) if c], key=lambda item: len(item[1]), reverse=True)
            for bucket_id, counter in used_buckets[:5]: print(f"  Bucket [{bucket_id:4d}]: {counter.most_common(top_k)}")
    print("--- End of Bucket Analysis ---\n")
    model.train()

@torch.no_grad()
def generate(model, tokenizer, device, prompt, max_new_tokens=50):
    model.eval()
    prompt_tokens = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    for _ in range(max_new_tokens):
        idx_cond = prompt_tokens[:, -model.config.block_size:]
        output = model(idx_cond)
        logits = output['logits']
        probs = F.softmax(logits[:, -1, :], dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        prompt_tokens = torch.cat((prompt_tokens, next_token), dim=1)
    decoded = tokenizer.decode(prompt_tokens.squeeze().tolist())
    print(f"--- PROMPT: '{prompt}' ---\nGENERATION: {decoded}\n-------------------------------------")
    model.train()
    
def chat(checkpoint_path, device, max_new_tokens=256):
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    tokenizer = tiktoken.get_encoding("cl100k_base"); config.vocab_size = tokenizer.n_vocab
    model = DiscretizedManifoldTransformer(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"\nModel loaded ({sum(p.numel() for p in model.parameters())/1e6:.2f}M params). Chat away! (type 'exit' to quit)")
    while True:
        prompt = input("You: ");
        if prompt.lower() in ['exit', 'quit']: break
        prompt_tokens = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
        print("Model:", end=" ", flush=True)
        for _ in range(max_new_tokens):
            idx_cond = prompt_tokens[:, -model.config.block_size:]
            with torch.no_grad(): 
                output = model(idx_cond)
                logits = output['logits']
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            if hasattr(tokenizer, 'eot_token') and next_token.item() == tokenizer.eot_token: break
            prompt_tokens = torch.cat((prompt_tokens, next_token), dim=1)
            print(tokenizer.decode([next_token.item()]), end="", flush=True)
        print("\n")

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

def main():
    parser = argparse.ArgumentParser(description="Train or chat with a Discretized Manifold Transformer.")
    parser.add_argument('--parquet_folder', type=str, default=None, help='Path to a folder of parquet files for training.')
    parser.add_argument('--text_column', type=str, default='text', help='The name of the text column in the parquet files.')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'nobackprop'], help='Optimizer to use for training.')
    args = parser.parse_args()
    torch.set_float32_matmul_precision('high')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    BATCH_SIZE = 4
    GRAD_ACCUM_STEPS = 8
    LEARNING_RATE = 3e-4
    EVAL_EVERY = 20000
    CHECKPOINT_DIR = "checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    tokenizer = tiktoken.get_encoding("p50k_base")
    config = ModelConfig()
    config.vocab_size = tokenizer.n_vocab
    model = DiscretizedManifoldTransformer(config).to(device)
    optimizer = None
    scaler = None
    if args.optimizer == 'nobackprop':
        print("Using pure NoBackprop optimizer with explicit reverse pass.")
        optimizer = NoBackpropOptimizer(model, learning_rate=1e-5)
    else:
        print("Using standard AdamW optimizer.")
        main_params = [p for name, p in model.named_parameters() if 'vq' not in name]
        vq_params = [p for name, p in model.named_parameters() if 'vq' in name]
        optimizer_main = torch.optim.AdamW(main_params, lr=LEARNING_RATE, betas=(0.9, 0.95))
        optimizer_vq = torch.optim.AdamW(vq_params, lr=LEARNING_RATE / 10.0, betas=(0.9, 0.95))
        optimizer = (optimizer_main, optimizer_vq)
        scaler = torch.cuda.amp.GradScaler(enabled=(device == 'cuda'))
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters.")
    print(f"Effective batch size: {BATCH_SIZE * GRAD_ACCUM_STEPS}")
    if not args.parquet_folder:
        print("No training data provided. Exiting.")
        return
    if pd is None: raise ImportError("Pandas is not installed. Please run 'pip install pandas pyarrow'.")
    parquet_files = sorted(glob.glob(os.path.join(args.parquet_folder, '*.parquet')))
    if not parquet_files: raise ValueError(f"No .parquet files found in {args.parquet_folder}")
    print(f"Found {len(parquet_files)} parquet files for training.")
    dataset = TokenStreamDataset(parquet_files, tokenizer, args.text_column, config.block_size)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0)
    model.train()
    global_step = 0
    for micro_step, batch_chunk in enumerate(dataloader):
        xb = batch_chunk[:, :config.block_size].to(device, non_blocking=True)
        yb = batch_chunk[:, 1:config.block_size+1].to(device, non_blocking=True)
        if args.optimizer == 'nobackprop':
            losses = optimizer.step(xb, yb)
            if (micro_step + 1) % GRAD_ACCUM_STEPS == 0:
                global_step += 1
                if global_step % 100 == 0 and losses:
                    print(f"Step {global_step} | Losses -> CE: {losses['ce_loss']:.4f}, Q: {losses['q_loss']:.4f}, Conf: {losses['conf_loss']:.4f}")
        else: # Standard AdamW training path
            optimizer_main, optimizer_vq = optimizer
            with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
                model_output = model(xb, yb)
                losses = model_output['losses']
                main_loss = (losses['ce_loss'] + losses['conf_loss'] * config.confidence_loss_weight) / GRAD_ACCUM_STEPS
                vq_loss = (losses['q_loss'] + losses['e_loss'] * config.commitment_cost) / GRAD_ACCUM_STEPS
            scaler.scale(main_loss).backward(retain_graph=True)
            scaler.scale(vq_loss).backward()
            if (micro_step + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(optimizer_main); scaler.unscale_(optimizer_vq)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer_main); scaler.step(optimizer_vq)
                scaler.update()
                optimizer_main.zero_grad(set_to_none=True); optimizer_vq.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % 100 == 0:
                    print(f"Step {global_step} | Losses -> CE: {losses['ce_loss']:.4f}, Conf: {losses['conf_loss']:.4f}, Q: {losses['q_loss']:.4f}")

        if global_step > 0 and global_step % EVAL_EVERY == 0:
            generate(model, tokenizer, device, prompt="The meaning of life is")

    print("\n--- Training Finished ---")

if __name__ == '__main__':
    main()