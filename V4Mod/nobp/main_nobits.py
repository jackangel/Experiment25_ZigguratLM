# --- START OF FILE main.py ---

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

from torch.utils.data import IterableDataset, DataLoader
import pyarrow.parquet as pq

try:
    import pandas as pd
except ImportError:
    pd = None

from model import ModelConfig, DiscretizedManifoldTransformer

# --- Utility and Inference functions (generate, chat, etc.) ---
@torch.no_grad()
def display_bucket_contents(model, tokenizer, device, analysis_tokens, stage_to_show=0, layer_to_show=0, top_k=8):
    model.eval()
    print(f"\n--- BUCKET ANALYSIS (Stage {stage_to_show}, Layer {layer_to_show}) ---")
    tokens = analysis_tokens.unsqueeze(0).to(device)
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
            for bucket_id, counter in used_buckets[:5]: print(f"  Bucket [{bucket_id:4d}] (Unique: {len(counter):,d}): {counter.most_common(top_k)}")
    print("--- End of Bucket Analysis ---\n")
    model.train()

@torch.no_grad()
def generate(model, tokenizer, device, prompt, max_new_tokens=50):
    model.eval()
    prompt_tokens = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    for _ in range(max_new_tokens):
        idx_cond = prompt_tokens[:, -model.config.block_size:]
        logits, _ = model(idx_cond)
        logits, probs = logits[:, -1, :], F.softmax(logits[:, -1, :], dim=-1)
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
            with torch.no_grad(): logits, _ = model(idx_cond)
            logits, probs = logits[:, -1, :], F.softmax(logits[:, -1, :], dim=-1)
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

if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    parser = argparse.ArgumentParser(description="Train DMT with local NoBackprop objectives.")
    parser.add_argument('--parquet_folder', type=str, default=None, help='Path to a folder of parquet files for training.')
    parser.add_argument('--text_column', type=str, default='text', help='The name of the text column in the parquet files.')
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    BATCH_SIZE = 6
    EVAL_EVERY = 20000
    CHECKPOINT_DIR = "checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    tokenizer = tiktoken.get_encoding("p50k_base")
    config = ModelConfig()
    config.vocab_size = tokenizer.n_vocab
    model = DiscretizedManifoldTransformer(config).to(device)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters.")
    
    # Optimizer for the MAIN predictive path (embeddings, stage 1, heads)
    main_path_params = (
        list(model.wte.parameters()) +
        list(model.rope.parameters()) +
        list(model.controller.parameters()) +
        list(model.transformation_ops.parameters()) +
        list(model.stages[0].parameters()) +
        list(model.ln_f.parameters()) +
        list(model.lm_head.parameters())
    )
    if config.use_confidence_circuit:
        main_path_params += list(model.confidence_head.parameters())

    final_optimizer = torch.optim.AdamW(main_path_params, lr=3e-4, betas=(0.9, 0.95))

    if args.parquet_folder:
        if pd is None: raise ImportError("Pandas is not installed. Please run 'pip install pandas pyarrow'.")
        parquet_files = sorted(glob.glob(os.path.join(args.parquet_folder, '*.parquet')))
        if not parquet_files: raise ValueError(f"No .parquet files found in {args.parquet_folder}")
        
        dataset = TokenStreamDataset(parquet_files, tokenizer, args.text_column, config.block_size)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
        
        model.train()
        global_step = 0
        for batch_chunk in dataloader:
            xb = batch_chunk[:, :config.block_size].to(device, non_blocking=True)
            yb = batch_chunk[:, 1:config.block_size+1].to(device, non_blocking=True)
            global_step += 1

            # Lists to store losses from stage 1 for logging and backprop
            stage1_recon_losses = []
            stage1_q_losses = []
            
            # --- HYBRID TRAINING ORCHESTRATOR ---
            with torch.cuda.amp.autocast():
                # 1. Forward through the main path (embeddings + stage 1)
                base_emb = model.wte(xb)
                abstraction_logits = model.controller(base_emb)
                abstraction_weights = F.softmax(abstraction_logits, dim=-1)
                rolled_embeddings = [base_emb] + [op(base_emb) for op in model.transformation_ops]
                stacked_embeddings = torch.stack(rolled_embeddings, dim=0)
                weights = abstraction_weights.permute(2, 0, 1).unsqueeze(-1)
                tok_emb = (stacked_embeddings * weights).sum(dim=0)
                x = model.drop(tok_emb + model.rope(tok_emb))

                # --- Stage 1 is part of main path: no detach, collect auxiliary losses ---
                stage1_blocks = model.stages[0]
                for block in stage1_blocks:
                    x_input = x
                    x, q_loss, e_loss, _ = block(x_input, return_indices=True)
                    x_reconstructed = block.local_reconstruction_head(x)
                    
                    stage1_recon_losses.append(F.mse_loss(x_reconstructed, x_input))
                    stage1_q_losses.append(q_loss + e_loss)
                
                output_from_stage1 = x

                # 2. Final prediction and CE loss
                final_hidden_state = model.ln_f(output_from_stage1)
                logits = model.lm_head(final_hidden_state)
                ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1), ignore_index=-1)
                
                # 3. Confidence Loss Calculation
                conf_loss = torch.tensor(0.0, device=device)
                if config.use_confidence_circuit:
                    confidence_logits = model.confidence_head(final_hidden_state.detach()).squeeze(-1)
                    with torch.no_grad():
                        predicted_tokens = logits.argmax(dim=-1)
                        correctness_targets = (predicted_tokens == yb).float()
                    conf_loss = model.bce_loss_fn(confidence_logits, correctness_targets)

                # 4. Combine all losses for the main predictive path
                total_aux_loss = sum(stage1_recon_losses) / len(stage1_recon_losses) + sum(stage1_q_losses) / len(stage1_q_losses)
                total_main_loss = ce_loss + total_aux_loss + (conf_loss * config.confidence_loss_weight)

            # 5. Backward pass for the main path
            final_optimizer.zero_grad()
            total_main_loss.backward()
            final_optimizer.step()

            # --- 6. Decoupled local updates for stages 2 and beyond ---
            x = output_from_stage1.detach() # Detach here to create a new graph starting point
            
            # Initialize variables for logging in case loops don't run
            last_decoupled_q_loss = torch.tensor(0.0)
            upsampling_loss = torch.tensor(0.0)

            for i in range(1, len(model.stages)):
                stage = model.stages[i]
                for block in stage:
                    x_input = x.detach().requires_grad_(True)
                    with torch.cuda.amp.autocast():
                        x_output, q_loss, e_loss, _ = block(x_input, return_indices=True)
                        last_decoupled_q_loss = q_loss + e_loss # Capture for logging
                        x_reconstructed = block.local_reconstruction_head(x_output)
                        local_loss = F.mse_loss(x_reconstructed, x_input) + last_decoupled_q_loss
                    
                    block.local_optimizer.zero_grad()
                    local_loss.backward()
                    block.local_optimizer.step()
                    x = x_output

                if x.shape[1] <= 1: break # Stop if sequence is too short
                
                # Note: The fusion block index is i-1 because there is one less fusion block than stages
                if i-1 < len(model.fusion_blocks):
                    fusion_block = model.fusion_blocks[i-1]
                    x_input_long = x.detach().requires_grad_(True)
                    with torch.cuda.amp.autocast():
                        x_output_short = fusion_block(x_input_long)
                        x_reconstructed_long = fusion_block.local_upsampling_head(x_output_short)
                        upsampling_loss = F.mse_loss(x_reconstructed_long[:, :x_input_long.shape[1], :], x_input_long)

                    fusion_block.local_optimizer.zero_grad()
                    upsampling_loss.backward()
                    fusion_block.local_optimizer.step()
                    x = x_output_short

            # --- 7. Comprehensive Reporting Block ---
            if global_step % 100 == 0:
                avg_stage1_q_loss = (sum(stage1_q_losses) / len(stage1_q_losses)).item()
                avg_stage1_recon_loss = (sum(stage1_recon_losses) / len(stage1_recon_losses)).item()

                print(
                    f"Step {global_step} | "
                    f"CE: {ce_loss.item():.4f} | "
                    f"Confidence: {conf_loss.item():.4f} | "
                    f"S1 Avg Q-Loss: {avg_stage1_q_loss:.4f} | "
                    f"S1 Avg Recon: {avg_stage1_recon_loss:.4f} | "
                    f"Last Decoupled Q-Loss: {last_decoupled_q_loss.item():.4f}"
                )

            if global_step > 0 and global_step % EVAL_EVERY == 0:
                generate(model, tokenizer, device, prompt="The meaning of life is")
                analysis_tokens_str = "This is a sample text for analyzing the model's internal state."
                analysis_sample = torch.tensor(tokenizer.encode(analysis_tokens_str), dtype=torch.long)[:config.block_size]
                display_bucket_contents(model, tokenizer, device, analysis_sample, stage_to_show=0, layer_to_show=0)
    
    print("\n--- Training Finished ---")