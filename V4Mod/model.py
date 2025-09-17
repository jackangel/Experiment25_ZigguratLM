# model.py

import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
import math

@dataclass
class ModelConfig:
    """Configuration for the Discretized Manifold Transformer."""
    block_size: int = 256
    n_embd: int = 384
    n_stages: int = 12
    n_layer_per_stage: int = 3
    n_head: int = 6
    dropout: float = 0.1
    num_codes: int = 1024
    vq_levels: int = 4
    commitment_cost: float = 0.25
    vocab_size: int = None
    chunk_size: int = 256
    n_levels: int = 4
    use_scratchpad: bool = True
    scratchpad_dim: int = 256
    use_deliberator: bool = True
    deliberation_steps: int = 2
    use_confidence_circuit: bool = True
    confidence_loss_weight: float = 0.2

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
        self.num_heads, self.head_size, self.chunk_size = num_heads, embedding_dim // num_heads, local_chunk_size
        self.mark_net, self.gate_net = nn.Linear(embedding_dim, embedding_dim), nn.Linear(embedding_dim, embedding_dim)
        self.card_norm, self.carry_norm = nn.LayerNorm(self.head_size), nn.LayerNorm(self.head_size)
        self.head_output_net = nn.Sequential(nn.Linear(self.head_size * 2, self.head_size * 2), nn.GELU(), nn.Linear(self.head_size * 2, self.head_size))
        self.proj, self.ln = nn.Linear(embedding_dim, embedding_dim), nn.LayerNorm(embedding_dim)
        torch.nn.init.zeros_(self.proj.weight); torch.nn.init.zeros_(self.proj.bias)
    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        pad_len = (self.chunk_size - T % self.chunk_size) % self.chunk_size
        if pad_len > 0: x_padded = F.pad(x, (0, 0, 0, pad_len)); T_padded = x_padded.shape[1]
        else: x_padded, T_padded = x, T
        num_chunks, H, D = T_padded // self.chunk_size, self.num_heads, self.head_size
        potential_marks, gates = self.mark_net(x_padded), torch.sigmoid(self.gate_net(x_padded))
        def reshape_for_chunks(tensor): return tensor.view(B, num_chunks, self.chunk_size, H, D).permute(0, 3, 1, 2, 4)
        x_heads, potential_marks, gates = reshape_for_chunks(x_padded), reshape_for_chunks(potential_marks), reshape_for_chunks(gates)
        gated_marks = gates * potential_marks
        local_cumulative_marks = torch.cumsum(gated_marks, dim=3)
        chunk_sums = local_cumulative_marks[:, :, :, -1, :].clone()
        initial_carry = torch.zeros(B, H, 1, D, device=x.device, dtype=x.dtype)
        carry_values_intermediate = torch.cumsum(chunk_sums, dim=2)
        carries_for_each_chunk = torch.cat([initial_carry, carry_values_intermediate[:, :, :-1, :]], dim=2)
        normalized_carries = self.carry_norm(carries_for_each_chunk).unsqueeze(3)
        marks_with_carry = local_cumulative_marks + normalized_carries
        initial_card = normalized_carries
        cards_passed_local = torch.cat([initial_card, marks_with_carry[:, :, :, :-1, :]], dim=3)
        cards_passed = self.card_norm(cards_passed_local)
        combined_head_info = torch.cat([x_heads, cards_passed], dim=-1)
        head_output = self.head_output_net(combined_head_info)
        out = head_output.permute(0, 2, 3, 1, 4).contiguous().view(B, T_padded, C)
        out = self.proj(out)
        if pad_len > 0: out = out[:, :-pad_len, :]
        return x[:, :out.shape[1], :] + self.ln(out)

class VectorQuantizer(nn.Module):
    def __init__(self, num_codes: int, embedding_dim: int, commitment_cost: float):
        super().__init__(); self.embedding_dim, self.num_codes, self.commitment_cost = embedding_dim, num_codes, commitment_cost
        self.embedding = nn.Embedding(self.num_codes, self.embedding_dim); self.embedding.weight.data.uniform_(-1.0 / self.num_codes, 1.0 / self.num_codes)
    def forward(self, inputs):
        flat_input = inputs.view(-1, self.embedding_dim)
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) + torch.sum(self.embedding.weight**2, dim=1) - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
        encoding_indices = torch.argmin(distances, dim=1)
        quantized = self.embedding(encoding_indices).view(inputs.shape)
        q_latent_loss, e_latent_loss = F.mse_loss(quantized, inputs.detach()), F.mse_loss(quantized.detach(), inputs)
        quantized_out = inputs + (quantized - inputs).detach()
        return quantized_out, q_latent_loss, e_latent_loss, quantized, encoding_indices.view(inputs.shape[:-1])

class ResidualVectorQuantizer(nn.Module):
    def __init__(self, num_levels: int, num_codes: int, embedding_dim: int, commitment_cost: float):
        super().__init__(); self.num_levels = num_levels
        self.quantizers = nn.ModuleList([VectorQuantizer(num_codes, embedding_dim, commitment_cost) for _ in range(num_levels)])
    def forward(self, inputs):
        residual, total_quantized_output, all_indices = inputs, torch.zeros_like(inputs), []
        total_q_loss, total_e_loss = 0.0, 0.0
        for quantizer in self.quantizers:
            _, q_loss, e_loss, quantized_pure, indices = quantizer(residual)
            all_indices.append(indices); total_quantized_output += quantized_pure
            total_q_loss, total_e_loss = total_q_loss + q_loss, total_e_loss + e_loss
            residual = residual - quantized_pure.detach()
        final_quantized_ste = inputs + (total_quantized_output - inputs).detach()
        return final_quantized_ste, total_q_loss/self.num_levels, total_e_loss/self.num_levels, torch.stack(all_indices, dim=0)

class DiscretizedManifoldBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attention = ChunkedMultiHeadCardPassingLayer(num_heads=config.n_head, embedding_dim=config.n_embd, local_chunk_size=config.chunk_size)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(nn.Linear(config.n_embd, 4*config.n_embd), nn.GELU(), nn.Linear(4*config.n_embd, config.n_embd), nn.Dropout(config.dropout))
        self.vq = ResidualVectorQuantizer(config.vq_levels, config.num_codes, config.n_embd, config.commitment_cost)
        self.ln_3 = nn.LayerNorm(config.n_embd)
        if config.use_scratchpad:
            self.scratchpad_gru = nn.GRU(config.n_embd, config.scratchpad_dim, batch_first=True)
            self.scratchpad_proj = nn.Linear(config.scratchpad_dim, config.n_embd)
    def forward(self, x, return_indices=False):
        if self.config.use_deliberator:
            for _ in range(self.config.deliberation_steps): x = x + self.attention(self.ln_1(x))
        else: x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        if self.config.use_scratchpad and hasattr(self, 'scratchpad_gru'):
            scratchpad_out, _ = self.scratchpad_gru(x); x = x + self.scratchpad_proj(scratchpad_out)
        x_norm = self.ln_3(x)
        quantized_x, q_loss, e_loss, indices = self.vq(x_norm)
        output = x + quantized_x
        if return_indices: return output, q_loss, e_loss, indices
        return output, q_loss, e_loss

class ResidualTokenFusionBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__(); self.fusion_conv = nn.Conv1d(config.n_embd, config.n_embd, kernel_size=2, stride=2)
        self.fusion_ln = nn.LayerNorm(config.n_embd); self.skip_pool = nn.AvgPool1d(kernel_size=2, stride=2)
        self.activation = nn.GELU(); self.final_ln = nn.LayerNorm(config.n_embd)
    def forward(self, x):
        x_permuted = x.permute(0, 2, 1)
        x_fused = self.fusion_ln(self.fusion_conv(x_permuted).permute(0, 2, 1))
        x_skip = self.skip_pool(x_permuted).permute(0, 2, 1)
        return self.final_ln(self.activation(x_fused + x_skip))

class ScaledSinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
    def forward(self, x, scale: int = 1):
        S = x.size(1)
        position = (torch.arange(0, S, device=x.device).float() * scale).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model)).to(x.device)
        pe = torch.zeros(S, self.d_model, device=x.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return x + pe.unsqueeze(0)

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
        self.transformation_ops = nn.ModuleList([nn.Linear(config.n_embd, config.n_embd, bias=False) for _ in range(config.n_levels - 1)])
        self.controller = nn.Sequential(nn.Linear(config.n_embd, config.n_embd // 2), nn.ReLU(), nn.Linear(config.n_embd // 2, config.n_levels))
        self.scaled_pe = ScaledSinusoidalPositionalEncoding(config.n_embd)
        if config.use_confidence_circuit:
            self.confidence_head = nn.Sequential(nn.Linear(config.n_embd, config.n_embd // 4), nn.GELU(), nn.Linear(config.n_embd // 4, 1))
            self.bce_loss_fn = nn.BCEWithLogitsLoss()
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None: torch.nn.init.zeros_(module.bias)
        total_layers = self.config.n_stages * self.config.n_layer_per_stage
        if hasattr(module, 'mlp') and isinstance(module.mlp[-2], nn.Linear): torch.nn.init.normal_(module.mlp[-2].weight, mean=0.0, std=0.02/math.sqrt(2*total_layers))

    def forward(self, idx, targets=None, return_indices=False):
        base_emb = self.wte(idx)
        abstraction_logits = self.controller(base_emb)
        abstraction_weights = F.softmax(abstraction_logits, dim=-1)
        rolled_embeddings = [base_emb] + [op(base_emb) for op in self.transformation_ops]
        stacked_embeddings = torch.stack(rolled_embeddings, dim=0)
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
                else: x, q_loss, e_loss = block(x)
                total_q_loss, total_e_loss = total_q_loss + q_loss, total_e_loss + e_loss
            if return_indices: all_stage_indices.append(torch.stack(stage_indices))
            if i == 0: output_from_stage1 = x
            if i < self.config.n_stages - 1:
                if x.shape[1] > 1:
                    x = self.fusion_blocks[i](x)
                    x = self.scaled_pe(x, scale=2**(i + 1))
                else: break
        final_hidden_state = self.ln_f(output_from_stage1)
        logits = self.lm_head(final_hidden_state)
        if return_indices: return logits, all_stage_indices
        if targets is not None:
            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            total_layers = self.config.n_stages * self.config.n_layer_per_stage
            conf_loss = torch.tensor(0.0, device=logits.device)
            if self.config.use_confidence_circuit and hasattr(self, 'confidence_head'):
                confidence_logits = self.confidence_head(final_hidden_state.detach()).squeeze(-1)
                with torch.no_grad():
                    predicted_tokens = logits.argmax(dim=-1)
                    correctness_targets = (predicted_tokens == targets).float()
                conf_loss = self.bce_loss_fn(confidence_logits, correctness_targets)
            return logits, (ce_loss, total_q_loss / total_layers, total_e_loss / total_layers, conf_loss)
        return logits, None