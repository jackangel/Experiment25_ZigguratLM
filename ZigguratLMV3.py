import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
import tiktoken
import os
import math
import random
import argparse
from collections import Counter

@dataclass
class ModelConfig:
    block_size: int = 256
    n_embd: int = 384
    n_stages: int = 2
    n_layer_per_stage: int = 3
    n_head: int = 6
    dropout: float = 0.1
    num_codes: int = 1024
    vq_levels: int = 4
    commitment_cost: float = 0.25 # This is now only used for logging, not training
    vocab_size: int = None
    chunk_size: int = 256 # Added for the new attention mechanism
    # --- START of ENHANCEMENT ---
    # Added n_levels for the Rolling Embedding Space (RES) mechanism
    # as seen in https://github.com/jackangel/Experiment35_Roll777LM/blob/main/res.py
    n_levels: int = 4 # Number of embedding spaces to mix. Level 0 is the base.
    # --- END of ENHANCEMENT ---


class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, dim, max_seq_len=256):
        super().__init__()
        assert dim % 2 == 0
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len, device=inv_freq.device).type_as(inv_freq)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        self.register_buffer("cos_cached", freqs.cos()[None, :, :])
        self.register_buffer("sin_cached", freqs.sin()[None, :, :])

    def forward(self, x):
        seq_len = x.shape[1]
        cos = self.cos_cached[:, :seq_len, ...]; sin = self.sin_cached[:, :seq_len, ...]
        x1 = x[..., 0::2]; x2 = x[..., 1::2]
        rotated_x = torch.cat(((-x2 * sin) + (x1 * cos), (x1 * sin) + (x2 * cos)), dim=-1)
        return rotated_x

class ChunkedMultiHeadCardPassingLayer(nn.Module):
    def __init__(self, num_heads, embedding_dim, local_chunk_size):
        super().__init__()
        assert embedding_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_size = embedding_dim // num_heads
        self.chunk_size = local_chunk_size
        
        self.mark_net = nn.Linear(embedding_dim, embedding_dim)
        self.gate_net = nn.Linear(embedding_dim, embedding_dim)
        self.card_norm = nn.LayerNorm(self.head_size)
        self.carry_norm = nn.LayerNorm(self.head_size)

        self.head_output_net = nn.Sequential(
            nn.Linear(self.head_size * 2, self.head_size * 2),
            nn.GELU(),
            nn.Linear(self.head_size * 2, self.head_size)
        )
        self.proj = nn.Linear(embedding_dim, embedding_dim)
        self.ln = nn.LayerNorm(embedding_dim)

        torch.nn.init.zeros_(self.proj.weight)
        torch.nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        # This assertion is removed to allow for sequences that are not divisible by the chunk size.
        # assert T % self.chunk_size == 0, "Sequence length must be divisible by chunk size"
        
        # Pad the sequence to be divisible by the chunk size
        pad_len = (self.chunk_size - T % self.chunk_size) % self.chunk_size
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
            T = x.shape[1] # Update sequence length
            
        num_chunks = T // self.chunk_size
        H = self.num_heads
        D = self.head_size

        # --- 1. Initial Projections (Parallel across all tokens) ---
        potential_marks = self.mark_net(x)
        gates = torch.sigmoid(self.gate_net(x))
        
        # --- 2. Reshape for Chunked Processing ---
        # Reshape from (B, T, C) to (B, H, N, S, D)
        # B=Batch, H=Heads, N=Num Chunks, S=Chunk Size, D=Head Dim
        def reshape_for_chunks(tensor):
            return tensor.view(B, num_chunks, self.chunk_size, H, D).permute(0, 3, 1, 2, 4)

        x_heads = reshape_for_chunks(x)
        potential_marks = reshape_for_chunks(potential_marks)
        gates = reshape_for_chunks(gates)
        
        gated_marks = gates * potential_marks # Shape: (B, H, N, S, D)

        # --- THE PARALLEL SCAN REPLACEMENT ---
        
        # --- Step 1: Intra-Chunk Scan ---
        # Calculate cumulative sum *within* each chunk in parallel.
        # The cumsum is performed on the 'S' dimension (dim=3).
        local_cumulative_marks = torch.cumsum(gated_marks, dim=3) # Shape: (B, H, N, S, D)

        # Get the total sum of each chunk by taking the last element of the cumulative sum.
        # This will be used to calculate the carry between chunks.
        chunk_sums = local_cumulative_marks[:, :, :, -1, :].clone() # Shape: (B, H, N, D)
        
        # --- Step 2: Inter-Chunk Scan (Calculating the carry) ---
        # We need the carry *entering* each chunk. This is the cumulative sum of all *previous* chunk totals.
        # We can achieve this by doing a cumsum and then shifting the results right.
        initial_carry = torch.zeros(B, H, 1, D, device=x.device, dtype=x.dtype)
        
        # Calculate cumulative sum across the 'N' dimension (dim=2).
        carry_values_intermediate = torch.cumsum(chunk_sums, dim=2) # Shape: (B, H, N, D)
        
        # Concatenate initial zero carry and slice to get the correct carry for each chunk.
        # This is the "shift right" operation.
        carries_for_each_chunk = torch.cat([initial_carry, carry_values_intermediate[:, :, :-1, :]], dim=2) # Shape: (B, H, N, D)

        # Apply layer norm to the carries before they are used.
        # unsqueeze to add the 'S' dimension for broadcasting.
        normalized_carries = self.carry_norm(carries_for_each_chunk).unsqueeze(3) # Shape: (B, H, N, 1, D)
        
        # --- Step 3: Combine ---
        # Add the appropriate carry to every position within its corresponding chunk.
        # The `normalized_carries` tensor broadcasts across the 'S' dimension.
        marks_with_carry = local_cumulative_marks + normalized_carries # Shape: (B, H, N, S, D)

        # The card passed to a token is the cumulative sum of marks *before* it.
        # We achieve this by prepending an initial card and slicing off the last value.
        initial_card = normalized_carries # Use the chunk's carry as the initial card
        cards_passed_local = torch.cat([initial_card, marks_with_carry[:, :, :, :-1, :]], dim=3)
        cards_passed = self.card_norm(cards_passed_local) # Shape: (B, H, N, S, D)

        # --- 4. Final Head Output and Projection (Parallel) ---
        combined_head_info = torch.cat([x_heads, cards_passed], dim=-1)
        head_output = self.head_output_net(combined_head_info) # Shape: (B, H, N, S, D)

        # Reshape back to (B, T, C) for the next layer
        out = head_output.permute(0, 2, 3, 1, 4).contiguous().view(B, T, C)
        
        out = self.proj(out)
        
        # Unpad the output to match the original sequence length
        if pad_len > 0:
            out = out[:, :-pad_len, :]
            
        out = self.ln(out)
        return x[:, :out.shape[1], :] + out # Ensure x and out have the same sequence length before residual connection

class VectorQuantizer(nn.Module):
    def __init__(self, num_codes: int, embedding_dim: int, commitment_cost: float):
        super().__init__()
        self.embedding_dim, self.num_codes, self.commitment_cost = embedding_dim, num_codes, commitment_cost
        self.embedding = nn.Embedding(self.num_codes, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_codes, 1.0 / self.num_codes)
    def forward(self, inputs):
        flat_input = inputs.view(-1, self.embedding_dim)
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) + torch.sum(self.embedding.weight**2, dim=1) - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_codes).float()
        quantized = torch.matmul(encodings, self.embedding.weight).view(inputs.shape)
        q_latent_loss, e_latent_loss = F.mse_loss(quantized, inputs.detach()), F.mse_loss(quantized.detach(), inputs)
        quantized_out = inputs + (quantized - inputs).detach()
        return quantized_out, q_latent_loss, e_latent_loss, quantized, encoding_indices.view(inputs.shape[:-1])

class ResidualVectorQuantizer(nn.Module):
    def __init__(self, num_levels: int, num_codes: int, embedding_dim: int, commitment_cost: float):
        super().__init__()
        self.num_levels = num_levels
        self.quantizers = nn.ModuleList([VectorQuantizer(num_codes, embedding_dim, commitment_cost) for _ in range(num_levels)])
    def forward(self, inputs):
        residual, total_quantized_output, all_indices = inputs, torch.zeros_like(inputs), []
        total_q_loss, total_e_loss = 0.0, 0.0
        for quantizer in self.quantizers:
            _, q_loss, e_loss, quantized_pure, indices = quantizer(residual)
            all_indices.append(indices)
            total_quantized_output += quantized_pure
            total_q_loss, total_e_loss = total_q_loss + q_loss, total_e_loss + e_loss
            residual = residual - quantized_pure.detach()
        final_quantized_ste = inputs + (total_quantized_output - inputs).detach()
        return final_quantized_ste, total_q_loss/self.num_levels, total_e_loss/self.num_levels, torch.stack(all_indices, dim=0)

class DiscretizedManifoldBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attention = ChunkedMultiHeadCardPassingLayer(num_heads=config.n_head, embedding_dim=config.n_embd, local_chunk_size=config.chunk_size)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(nn.Linear(config.n_embd, 4*config.n_embd), nn.GELU(), nn.Linear(4*config.n_embd, config.n_embd), nn.Dropout(config.dropout))
        self.vq = ResidualVectorQuantizer(config.vq_levels, config.num_codes, config.n_embd, config.commitment_cost)
        self.ln_3 = nn.LayerNorm(config.n_embd)
    def forward(self, x, return_indices=False):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        x_norm = self.ln_3(x)
        quantized_x, q_loss, e_loss, indices = self.vq(x_norm)
        output = x + quantized_x
        if return_indices: return output, q_loss, e_loss, indices
        return output, q_loss, e_loss

class ResidualTokenFusionBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.fusion_conv = nn.Conv1d(config.n_embd, config.n_embd, kernel_size=2, stride=2)
        self.fusion_ln = nn.LayerNorm(config.n_embd)
        self.skip_pool = nn.AvgPool1d(kernel_size=2, stride=2)
        self.activation = nn.GELU()
        self.final_ln = nn.LayerNorm(config.n_embd)
    def forward(self, x):
        x_permuted = x.permute(0, 2, 1)
        x_fused = self.fusion_conv(x_permuted)
        x_fused = self.fusion_ln(x_fused.permute(0, 2, 1))
        x_skip = self.skip_pool(x_permuted)
        x_skip = x_skip.permute(0, 2, 1)
        output = self.activation(x_fused + x_skip)
        return self.final_ln(output)

class DiscretizedManifoldTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.rope = RotaryPositionalEmbeddings(config.n_embd, config.block_size)
        self.drop = nn.Dropout(config.dropout)
        self.stages, self.fusion_blocks = nn.ModuleList(), nn.ModuleList()
        for i in range(config.n_stages):
            self.stages.append(nn.ModuleList([DiscretizedManifoldBlock(config) for _ in range(config.n_layer_per_stage)]))
            if i < config.n_stages - 1: self.fusion_blocks.append(ResidualTokenFusionBlock(config))
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight
        
        # --- START of ENHANCEMENT ---
        # RES: Components for Rolling Embedding Space
        # The "rollers" - n_levels-1 because Level 0 is the identity (base embedding)
        self.transformation_ops = nn.ModuleList(
            [nn.Linear(config.n_embd, config.n_embd, bias=False) for _ in range(config.n_levels - 1)]
        )
        # The Abstraction Level Controller (operates on base embeddings)
        self.controller = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd // 2),
            nn.ReLU(),
            nn.Linear(config.n_embd // 2, config.n_levels)
        )
        # --- END of ENHANCEMENT ---
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None: torch.nn.init.zeros_(module.bias)
        total_layers = self.config.n_stages * self.config.n_layer_per_stage
        # The following check for SimplifiedRetention is removed as it's no longer used.
        # if isinstance(module, SimplifiedRetention): torch.nn.init.normal_(module.out_proj.weight, mean=0.0, std=0.02/math.sqrt(2*total_layers))
        if hasattr(module, 'mlp') and isinstance(module.mlp[-2], nn.Linear): torch.nn.init.normal_(module.mlp[-2].weight, mean=0.0, std=0.02/math.sqrt(2*total_layers))

    def forward(self, idx, targets=None, return_indices=False):
        base_emb = self.wte(idx)

        # 1. Controller predicts abstraction weights from the base embeddings
        abstraction_logits = self.controller(base_emb) # No need for detach() here anymore
        abstraction_weights = F.softmax(abstraction_logits, dim=-1)

        # 2. Generate all "rolled" embeddings
        rolled_embeddings = [base_emb]
        for op in self.transformation_ops:
            rolled_embeddings.append(op(base_emb))
        
        stacked_embeddings = torch.stack(rolled_embeddings, dim=0)

        # 3. Create Final Dynamic Embedding by soft-mixing
        weights = abstraction_weights.permute(2, 0, 1).unsqueeze(-1)
        tok_emb = (stacked_embeddings * weights).sum(dim=0)

        x = self.drop(tok_emb + self.rope(tok_emb))
        all_stage_indices, output_from_stage1 = [], None
        total_q_loss, total_e_loss = 0.0, 0.0
        for i, stage in enumerate(self.stages):
            stage_indices = []
            for block in stage:
                if return_indices:
                    x, q_loss, e_loss, indices = block(x, return_indices=True)
                    stage_indices.append(indices)
                else:
                    x, q_loss, e_loss = block(x)
                total_q_loss, total_e_loss = total_q_loss + q_loss, total_e_loss + e_loss
            if return_indices: all_stage_indices.append(torch.stack(stage_indices))
            if i == 0: output_from_stage1 = x
            if i < self.config.n_stages - 1:
                if x.shape[1] > 1: x = self.fusion_blocks[i](x)
                else: break
        logits = self.lm_head(self.ln_f(output_from_stage1))
        if return_indices: return logits, all_stage_indices
        if targets is not None:
            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            total_layers = self.config.n_stages * self.config.n_layer_per_stage
            return logits, (ce_loss, total_q_loss / total_layers, total_e_loss / total_layers)
        return logits, None
@torch.no_grad()
def display_bucket_contents(model, tokenizer, device, analysis_tokens, stage_to_show=0, layer_to_show=0, top_k=8):
    model.eval()
    print(f"\n--- BUCKET ANALYSIS (Stage {stage_to_show}, Layer {layer_to_show}) ---")
    tokens = analysis_tokens.unsqueeze(0).to(device)
    _, all_stage_indices = model(tokens, return_indices=True)
    config = model.config
    seq_len_multiplier = (0.5 ** stage_to_show)
    if stage_to_show >= len(all_stage_indices):
        print(f"Error: Stage {stage_to_show} not found in model output."); model.train(); return
    layer_indices = all_stage_indices[stage_to_show][layer_to_show]
    if stage_to_show > 0:
        print("--- Analyzing Multi-Token Concepts (Token labels not applicable) ---")
        for level_idx in range(config.vq_levels):
            level_counter = Counter(layer_indices[level_idx].flatten().tolist())
            print(f"\n--- VQ HIERARCHY LEVEL {level_idx} ---")
            print(f"  Top 5 most used buckets: {level_counter.most_common(5)}")
    else:
        bucket_contents = [[Counter() for _ in range(config.num_codes)] for _ in range(config.vq_levels)]
        expected_seq_len = int(config.block_size * seq_len_multiplier)
        for level_idx in range(config.vq_levels):
            for seq_idx in range(expected_seq_len):
                token_id = tokens[0, seq_idx].item()
                bucket_id = layer_indices[level_idx, 0, seq_idx].item()
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
    tokenizer = tiktoken.get_encoding("cl100k_base")
    config.vocab_size = tokenizer.n_vocab
    model = DiscretizedManifoldTransformer(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"\nModel loaded ({sum(p.numel() for p in model.parameters())/1e6:.2f}M params). Chat away!")
    while True:
        prompt = input("You: ")
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
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train or chat with a Multi-Stage Hierarchical DMT.")
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to a checkpoint for chat mode.')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if args.checkpoint:
        chat(args.checkpoint, device)
    else:
        print("Starting training...")
        NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE = 200, 6, 3e-4
        EVAL_EVERY = 500
        CHECKPOINT_DIR = "checkpoints"
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        
        tokenizer = tiktoken.get_encoding("cl100k_base")
        with open('input.txt', 'r', encoding='utf-8') as f: text = f.read()
        all_data = torch.tensor(tokenizer.encode(text))
        n = int(0.9 * len(all_data))
        train_data, val_data = all_data[:n], all_data[n:]
        
        config = ModelConfig()
        config.vocab_size = tokenizer.n_vocab
        model = DiscretizedManifoldTransformer(config).to(device)
        
        # Using Differential Learning Rates
        main_params = [p for name, p in model.named_parameters() if 'vq.quantizers' not in name]
        optimizer_main = torch.optim.AdamW(main_params, lr=LEARNING_RATE, betas=(0.9, 0.95))
        vq_params = [p for name, p in model.named_parameters() if 'vq.quantizers' in name]
        optimizer_vq = torch.optim.AdamW(vq_params, lr=LEARNING_RATE / 10.0, betas=(0.9, 0.95))
        print(f"Model initialized with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters.")
        print(f"Training with two optimizers: Main (LR: {LEARNING_RATE}) and VQ (LR: {LEARNING_RATE / 10.0}).")
        
        global_step = 0
        for epoch in range(NUM_EPOCHS):
            print(f"\n--- Starting Epoch {epoch+1}/{NUM_EPOCHS} ---")
            possible_starts = list(range(len(train_data) - config.block_size - 1))
            random.shuffle(possible_starts)
            
            for i in range(len(possible_starts) // BATCH_SIZE):
                start_indices = possible_starts[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
                xb = torch.stack([train_data[j:j+config.block_size] for j in start_indices]).to(device)
                yb = torch.stack([train_data[j+1:j+config.block_size+1] for j in start_indices]).to(device)
                logits, (ce_loss, q_loss, e_loss) = model(xb, yb)
                
                main_loss = ce_loss
                vq_optimizer_loss = q_loss
                optimizer_main.zero_grad(set_to_none=True)
                optimizer_vq.zero_grad(set_to_none=True)
                main_loss.backward(retain_graph=True)
                vq_optimizer_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer_main.step()
                optimizer_vq.step()
                
                if global_step % 100 == 0:
                    total_loss = main_loss + vq_optimizer_loss # For display purposes only
                    print(f"Epoch {epoch+1}, Step {global_step}: Loss {total_loss.item():.4f} (CE: {ce_loss.item():.4f}, Q_loss: {q_loss.item():.4f}, E_loss: {e_loss.item():.4f})")
                
                if global_step > 0 and global_step % EVAL_EVERY == 0:
                    generate(model, tokenizer, device, prompt="The meaning of life is")
                    analysis_sample = val_data[:config.block_size]
                    display_bucket_contents(model, tokenizer, device, analysis_sample, stage_to_show=0, layer_to_show=0)
                    if config.n_stages > 1:
                        display_bucket_contents(model, tokenizer, device, analysis_sample, stage_to_show=1, layer_to_show=0)
                
                global_step += 1
                
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"dmt_checkpoint_epoch_{epoch+1}.pth")
            print(f"Saving checkpoint to {checkpoint_path}")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_main_state_dict': optimizer_main.state_dict(),
                'optimizer_vq_state_dict': optimizer_vq.state_dict(),
                'config': config
            }, checkpoint_path)
        print("\n--- Training Finished ---")
