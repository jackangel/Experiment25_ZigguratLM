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
from collections import Counter, deque
import glob

from torch.utils.data import IterableDataset, DataLoader, TensorDataset
from torch.cuda.amp import GradScaler
import pyarrow.parquet as pq

try:
    import pandas as pd
except ImportError:
    pd = None

from model import ModelConfig, DiscretizedManifoldTransformer

class GradientNormTracker:
    """Tracks gradient norm statistics with moving averages."""
    
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.norm_history = deque(maxlen=window_size)
        self.moving_avg = 0.0
        self.moving_std = 0.0
        
    def update(self, grad_norm):
        self.norm_history.append(grad_norm)
        if len(self.norm_history) >= 10:
            norms = torch.tensor(list(self.norm_history))
            self.moving_avg = norms.mean().item()
            self.moving_std = norms.std().item()
        
    def get_adaptive_clip_value(self, base_clip=1.0):
        if self.moving_avg == 0.0:
            return base_clip
        return max(base_clip, self.moving_avg + 2.0 * self.moving_std)

class EmbeddingGradientInterceptor(torch.autograd.Function):
    """AGGRESSIVE: Much less restrictive embedding gradients."""
    
    @staticmethod
    def forward(ctx, x, weight, vocab_size):
        output = F.embedding(x, weight)
        ctx.save_for_backward(x, weight)
        ctx.vocab_size = vocab_size
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        vocab_size = ctx.vocab_size
        
        grad_input = grad_weight = None
        
        if ctx.needs_input_grad[0]:
            grad_input = None
        
        if ctx.needs_input_grad[1]:
            grad_weight = torch.zeros_like(weight)
            
            # MUCH LESS AGGRESSIVE: Near-normal scaling
            vocab_scale = 1.0 / math.log(max(vocab_size, 100))  # Less impact
            batch_scale = 1.0 / math.log(max(x.numel(), 10))    # Less impact
            final_scale = vocab_scale * batch_scale * 1.0       # 10x larger
            
            for i in range(x.numel()):
                token_id = x.view(-1)[i].item()
                if 0 <= token_id < vocab_size:
                    pos_grad = grad_output.view(-1, grad_output.size(-1))[i] * final_scale
                    pos_grad = torch.clamp(pos_grad, -1.0, 1.0)  # Much wider
                    grad_weight[token_id] += pos_grad
            
            grad_weight = torch.clamp(grad_weight, -5.0, 5.0)  # Much wider
        
        return grad_input, grad_weight, None

class LMHeadGradientInterceptor(torch.autograd.Function):
    """AGGRESSIVE: Much less restrictive LM head gradients."""
    
    @staticmethod
    def forward(ctx, x, weight, bias, vocab_size):
        output = F.linear(x, weight, bias)
        ctx.save_for_backward(x, weight, bias)
        ctx.vocab_size = vocab_size
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias = ctx.saved_tensors
        vocab_size = ctx.vocab_size
        
        target_dtype = weight.dtype
        grad_output = grad_output.to(target_dtype)
        x = x.to(target_dtype)
        
        # MUCH LESS AGGRESSIVE: Near-normal scaling
        vocab_scale = 1.0 / math.log(max(vocab_size, 100))
        seq_scale = 1.0 / math.log(max(grad_output.size(-2), 10))
        batch_scale = 1.0 / math.log(max(grad_output.size(0), 10))
        final_scale = vocab_scale * seq_scale * batch_scale * 0.5  # Much larger
        
        scaled_grad_output = grad_output * final_scale
        
        grad_input = grad_weight = grad_bias = None
        
        if ctx.needs_input_grad[0]:
            grad_input = scaled_grad_output.matmul(weight)
        
        if ctx.needs_input_grad[1]:
            if scaled_grad_output.dim() > 2:
                grad_output_flat = scaled_grad_output.view(-1, scaled_grad_output.size(-1))
                x_flat = x.view(-1, x.size(-1))
                grad_weight = grad_output_flat.transpose(-2, -1).matmul(x_flat)
            else:
                grad_weight = scaled_grad_output.transpose(-2, -1).matmul(x)
            
            # Much wider clamp
            grad_weight = torch.clamp(grad_weight, -1.0, 1.0)
        
        if bias is not None and ctx.needs_input_grad[3]:
            grad_bias = scaled_grad_output
            while grad_bias.dim() > 1:
                grad_bias = grad_bias.sum(0)
            
            # Much less aggressive bias scaling
            bias_scale = 0.5
            grad_bias = grad_bias * bias_scale
            grad_bias = torch.clamp(grad_bias, -0.5, 0.5)
        
        return grad_input, grad_weight, grad_bias, None

class StabilizedEmbedding(nn.Module):
    """Much less conservative embedding initialization."""
    
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        self.weight = nn.Parameter(torch.randn(vocab_size, embed_dim))
        
        # MUCH LESS CONSERVATIVE initialization
        std = 0.02 / math.sqrt(embed_dim)  # Standard transformer init
        nn.init.normal_(self.weight, 0.0, std)
        print(f"AGGRESSIVE embedding initialized: vocab_size={vocab_size}, embed_dim={embed_dim}, std={std:.8f}")
    
    def forward(self, x):
        return EmbeddingGradientInterceptor.apply(x, self.weight, self.vocab_size)

class StabilizedLMHead(nn.Module):
    """Much less conservative LM head initialization."""
    
    def __init__(self, embed_dim, vocab_size, bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        
        self.weight = nn.Parameter(torch.randn(vocab_size, embed_dim))
        self.bias = nn.Parameter(torch.randn(vocab_size)) if bias else None
        
        # MUCH LESS CONSERVATIVE initialization
        std = 0.02 / math.sqrt(embed_dim)  # Standard transformer init
        nn.init.normal_(self.weight, 0.0, std)
        
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        
        print(f"AGGRESSIVE LM head initialized: {embed_dim} -> {vocab_size}, std={std:.8f}")
    
    def forward(self, x):
        return LMHeadGradientInterceptor.apply(x, self.weight, self.bias, self.vocab_size)

class GradientMonitor:
    def __init__(self):
        self.gradient_stats = {}
        self.nan_inf_count = 0
        self.explosion_count = 0
        self.norm_trackers = {
            'gp_params': GradientNormTracker(),
            'standard_params': GradientNormTracker(),
            'vq_params': GradientNormTracker(),
            'embedding_params': GradientNormTracker(),
            'head_params': GradientNormTracker(),
            'transformation_ops': GradientNormTracker(),
            'layer_norm_bias': GradientNormTracker(),
            'lm_head_params': GradientNormTracker()
        }
        
    def log_gradients(self, model, step):
        """Log gradient statistics for all parameters."""
        total_norm = 0.0
        param_count = 0
        max_grad = 0.0
        min_grad = float('inf')
        
        grad_stats_by_type = {
            'gp_params': {'norms': [], 'max': 0, 'min': float('inf')},
            'standard_params': {'norms': [], 'max': 0, 'min': float('inf')},
            'vq_params': {'norms': [], 'max': 0, 'min': float('inf')},
            'embedding_params': {'norms': [], 'max': 0, 'min': float('inf')},
            'head_params': {'norms': [], 'max': 0, 'min': float('inf')},
            'transformation_ops': {'norms': [], 'max': 0, 'min': float('inf')},
            'layer_norm_bias': {'norms': [], 'max': 0, 'min': float('inf')},
            'lm_head_params': {'norms': [], 'max': 0, 'min': float('inf')}
        }
        
        extreme_params = []
        successful_stabilization = []
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm()
                param_max = param.grad.data.abs().max()
                param_min = param.grad.data.abs().min()
                
                total_norm += param_norm.item() ** 2
                param_count += 1
                max_grad = max(max_grad, param_max.item())
                min_grad = min(min_grad, param_min.item())
                
                if param_norm.item() > 500.0:  # Much higher threshold
                    extreme_params.append((name, param_norm.item()))
                
                # Track all vocabulary parameters
                if 'wte' in name or 'lm_head' in name:
                    successful_stabilization.append((name, param_norm.item()))
                
                param_type = self._get_param_type(name)
                grad_stats_by_type[param_type]['norms'].append(param_norm.item())
                grad_stats_by_type[param_type]['max'] = max(grad_stats_by_type[param_type]['max'], param_max.item())
                grad_stats_by_type[param_type]['min'] = min(grad_stats_by_type[param_type]['min'], param_min.item())
                
                if param_type in self.norm_trackers:
                    self.norm_trackers[param_type].update(param_norm.item())
                
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    print(f"⚠️  NaN/Inf detected in {name} at step {step}")
                    self.nan_inf_count += 1
                
                if param_norm.item() > 500.0:  # Much higher threshold
                    self.explosion_count += 1
        
        total_norm = math.sqrt(total_norm)
        
        if 'total' not in self.norm_trackers:
            self.norm_trackers['total'] = GradientNormTracker()
        self.norm_trackers['total'].update(total_norm)
        
        log_frequency = 1 if step < 10 else (5 if step < 50 else 10)
        if step % log_frequency == 0:
            print(f"\n--- GRADIENT STATS (Step {step}) ---")
            print(f"Total norm: {total_norm:.4f} | Max: {max_grad:.6f} | Min: {min_grad:.6f}")
            print(f"Moving avg total norm: {self.norm_trackers['total'].moving_avg:.4f}")
            
            for param_type, stats in grad_stats_by_type.items():
                if stats['norms']:
                    avg_norm = sum(stats['norms']) / len(stats['norms'])
                    moving_avg = self.norm_trackers[param_type].moving_avg
                    adaptive_clip = self.norm_trackers[param_type].get_adaptive_clip_value()
                    print(f"{param_type}: avg_norm={avg_norm:.4f}, moving_avg={moving_avg:.4f}, adaptive_clip={adaptive_clip:.4f}, count={len(stats['norms'])}")
            
            # Show vocabulary parameter gradients
            if successful_stabilization:
                print(f"📊 VOCABULARY GRADIENTS ({len(successful_stabilization)}):")
                for name, norm in successful_stabilization:
                    print(f"  {name}: {norm:.4f}")
            
            if extreme_params:
                extreme_params.sort(key=lambda x: x[1], reverse=True)
                print("Top 5 extreme gradients:")
                for name, norm in extreme_params[:5]:
                    print(f"  {name}: {norm:.4f}")
            
            if total_norm > 200.0:  # Much higher threshold
                print(f"🔥 HIGH GRADIENT NORM: {total_norm:.4f}")
            else:
                print(f"✅ GRADIENT NORM: {total_norm:.4f}")
                
            if self.explosion_count > 0:
                print(f"💥 EXPLOSIONS DETECTED: {self.explosion_count}")
            if self.nan_inf_count > 0:
                print(f"🚫 NaN/Inf COUNT: {self.nan_inf_count}")
            print("--- END GRADIENT STATS ---\n")
        
        return total_norm, max_grad
    
    def _get_param_type(self, param_name):
        if 'wte' in param_name or 'embedding' in param_name:
            return 'embedding_params'
        elif 'lm_head' in param_name:
            return 'lm_head_params'
        elif 'transformation_ops' in param_name:
            return 'transformation_ops'
        elif ('ln' in param_name or 'norm' in param_name) and 'bias' in param_name:
            return 'layer_norm_bias'
        elif 'head_output_net' in param_name or 'confidence_head' in param_name:
            return 'head_params'
        elif 'vq.quantizers' in param_name:
            return 'vq_params'
        elif any(layer_type in param_name for layer_type in ['gate_net', 'mark_net', 'proj']):
            return 'gp_params'
        else:
            return 'standard_params'

class GlobalPerturbationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, layer_id, stage_idx, num_stages):
        output = F.linear(x, weight, bias)
        ctx.save_for_backward(x, weight, bias)
        ctx.layer_id = layer_id
        ctx.stage_idx = stage_idx
        ctx.num_stages = num_stages
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias = ctx.saved_tensors
        layer_id = ctx.layer_id
        stage_idx = ctx.stage_idx
        num_stages = ctx.num_stages
        
        target_dtype = weight.dtype
        grad_output = grad_output.to(target_dtype)
        x = x.to(target_dtype)
        
        # MUCH LESS AGGRESSIVE GP scaling
        stage_damping = 1.0 / (1.0 + stage_idx * 0.05)  # Minimal damping
        depth_scaling = 1.0 / math.sqrt(num_stages)
        combined_scaling = stage_damping * depth_scaling * 1.5  # Higher than before
        
        scaled_grad_output = grad_output * combined_scaling
        
        grad_input = grad_weight = grad_bias = None
        
        if ctx.needs_input_grad[0]:
            grad_input = scaled_grad_output.matmul(weight)
        
        if ctx.needs_input_grad[1]:
            if scaled_grad_output.dim() > 2:
                grad_output_flat = scaled_grad_output.view(-1, scaled_grad_output.size(-1))
                x_flat = x.view(-1, x.size(-1))
                grad_weight = grad_output_flat.transpose(-2, -1).matmul(x_flat)
            else:
                grad_weight = scaled_grad_output.transpose(-2, -1).matmul(x)
        
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = scaled_grad_output
            while grad_bias.dim() > 1:
                grad_bias = grad_bias.sum(0)
            
            # Much less aggressive bias scaling
            bias_scaling = 0.5 / math.sqrt(num_stages)
            grad_bias = grad_bias * bias_scaling
            grad_bias = torch.clamp(grad_bias, -5.0, 5.0)  # Much wider
        
        return grad_input, grad_weight, grad_bias, None, None, None

class EnhancedGPLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, layer_id=0, stage_idx=0, num_stages=12):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.layer_id = layer_id
        self.stage_idx = stage_idx
        self.num_stages = num_stages
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features)) if bias else None
        
        # MUCH LESS CONSERVATIVE initialization
        base_std = math.sqrt(2.0 / in_features)
        stage_scale = 1.0 / math.sqrt(num_stages)
        final_std = base_std * stage_scale * 1.0  # Much less conservative
        
        nn.init.normal_(self.weight, 0.0, final_std)
        
        if self.bias is not None:
            bias_std = final_std * 0.1  # Much less conservative
            nn.init.normal_(self.bias, 0.0, bias_std)
    
    def forward(self, x):
        return GlobalPerturbationFunction.apply(x, self.weight, self.bias, self.layer_id, self.stage_idx, self.num_stages)

class HybridDiscretizedManifoldTransformer(DiscretizedManifoldTransformer):
    """AGGRESSIVE: Maximum learning optimization."""
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        self._apply_aggressive_minimal_stabilization()
        self._replace_embedding_with_stabilized_version()
        self._replace_lm_head_with_stabilized_version()
        self._convert_to_gp_layers()
        self._minimal_freezing_only()  # Minimal freezing
        
    def _replace_embedding_with_stabilized_version(self):
        """Replace with much less conservative version."""
        print("Replacing embedding with AGGRESSIVE version...")
        
        if hasattr(self, 'wte') and isinstance(self.wte, nn.Embedding):
            old_embedding = self.wte
            vocab_size = old_embedding.num_embeddings
            embed_dim = old_embedding.embedding_dim
            
            new_embedding = StabilizedEmbedding(vocab_size, embed_dim)
            
            with torch.no_grad():
                new_embedding.weight.data.copy_(old_embedding.weight.data)
            
            self.wte = new_embedding
            print(f"AGGRESSIVE embedding: {vocab_size} x {embed_dim}")
        else:
            print("No standard embedding found")
    
    def _replace_lm_head_with_stabilized_version(self):
        """Replace with much less conservative version."""
        print("Replacing LM head with AGGRESSIVE version...")
        
        if hasattr(self, 'lm_head') and isinstance(self.lm_head, nn.Linear):
            old_head = self.lm_head
            embed_dim = old_head.in_features
            vocab_size = old_head.out_features
            has_bias = old_head.bias is not None
            
            new_head = StabilizedLMHead(embed_dim, vocab_size, bias=has_bias)
            
            with torch.no_grad():
                new_head.weight.data.copy_(old_head.weight.data)
                if has_bias and new_head.bias is not None:
                    new_head.bias.data.copy_(old_head.bias.data)
            
            self.lm_head = new_head
            print(f"AGGRESSIVE LM head: {embed_dim} -> {vocab_size}")
        else:
            print("No standard LM head found")
    
    def _apply_aggressive_minimal_stabilization(self):
        """Minimal stabilization - only what's absolutely necessary."""
        print("Applying MINIMAL stabilization...")
        
        num_stages = len(self.stages)
        
        for name, param in self.named_parameters():
            with torch.no_grad():
                # Only stabilize the most problematic transformation ops
                if 'transformation_ops' in name and 'weight' in name:
                    if param.dim() >= 2:
                        fan_in = param.size(-1)
                        std = 0.1 / math.sqrt(num_stages * fan_in)  # Much less aggressive
                        nn.init.normal_(param, 0.0, std)
                        print(f"  Minimal: {name}: std={std:.8f}")
                
                elif 'transformation_ops' in name and 'bias' in name:
                    std = 0.01  # Small but learnable
                    nn.init.normal_(param, 0.0, std)
                    print(f"  Minimal: {name}: std={std:.8f}")
                
                # Only zero the most critical layer norm biases
                elif ('ln' in name or 'norm' in name) and 'bias' in name and 'stages.0.0.' in name:
                    nn.init.zeros_(param)
                    print(f"  Zeroed only critical: {name}")
                
                # Everything else gets near-standard initialization
                elif 'weight' in name and param.dim() >= 2:
                    fan_in = param.size(-1)
                    std = math.sqrt(2.0 / fan_in) * 0.8  # Near-standard
                    nn.init.normal_(param, 0.0, std)
        
        print("Minimal stabilization complete.")
    
    def _minimal_freezing_only(self):
        """Freeze absolutely nothing - let everything learn."""
        print("NO PARAMETER FREEZING - Full learning enabled!")
        # We're not freezing anything!
        
    def _convert_to_gp_layers(self):
        """Convert to much less conservative GP layers."""
        layer_id = 0
        num_stages = len(self.stages)
        
        for stage_idx, stage in enumerate(self.stages):
            for block_idx, block in enumerate(stage):
                if hasattr(block, 'attention'):
                    attention_module = block.attention
                    for attr_name in dir(attention_module):
                        if not attr_name.startswith('_'):
                            attr = getattr(attention_module, attr_name)
                            if isinstance(attr, nn.Linear):
                                new_layer = EnhancedGPLinear(
                                    attr.in_features, 
                                    attr.out_features, 
                                    bias=attr.bias is not None,
                                    layer_id=layer_id,
                                    stage_idx=stage_idx,
                                    num_stages=num_stages
                                )
                                with torch.no_grad():
                                    new_layer.weight.data.copy_(attr.weight.data)
                                    if new_layer.bias is not None:
                                        new_layer.bias.data.copy_(attr.bias.data)
                                
                                setattr(attention_module, attr_name, new_layer)
                                layer_id += 1
                
                if hasattr(block, 'mlp') and isinstance(block.mlp, nn.ModuleList):
                    if len(block.mlp) > 0 and isinstance(block.mlp[0], nn.Linear):
                        old_layer = block.mlp[0]
                        new_layer = EnhancedGPLinear(
                            old_layer.in_features, 
                            old_layer.out_features, 
                            bias=old_layer.bias is not None,
                            layer_id=layer_id,
                            stage_idx=stage_idx,
                            num_stages=num_stages
                        )
                        with torch.no_grad():
                            new_layer.weight.data.copy_(old_layer.weight.data)
                            if new_layer.bias is not None:
                                new_layer.bias.data.copy_(old_layer.bias.data)
                        
                        block.mlp[0] = new_layer
                        layer_id += 1
        
        print(f"Converted {layer_id} layers to AGGRESSIVE GP")

class LayerwiseLearningRateOptimizer:
    """MAXIMUM learning rates while maintaining basic stability."""
    
    def __init__(self, model, learning_rate=5e-4, vq_lr_multiplier=0.2, gp_lr_multiplier=0.3, weight_decay=1e-4):
        self.model = model
        self.base_learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        self.param_groups = {
            'embedding': [],
            'lm_head': [],
            'early_stages': [],
            'mid_stages': [],
            'late_stages': [],
            'heads': [],
            'gp_params': [],
            'vq_params': [],
            'bias_params': [],
            'transformation_ops': [],
            'layer_norm_params': []
        }
        
        num_stages = len(model.stages)
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                print(f"Skipping frozen parameter: {name}")
                continue
                
            if 'vq.quantizers' in name:
                self.param_groups['vq_params'].append(param)
            elif 'wte' in name or 'embedding' in name:
                self.param_groups['embedding'].append(param)
            elif 'lm_head' in name:
                self.param_groups['lm_head'].append(param)
            elif 'transformation_ops' in name:
                self.param_groups['transformation_ops'].append(param)
            elif ('ln' in name or 'norm' in name):
                self.param_groups['layer_norm_params'].append(param)
            elif 'head_output_net' in name or 'confidence_head' in name:
                self.param_groups['heads'].append(param)
            elif 'bias' in name:
                self.param_groups['bias_params'].append(param)
            else:
                is_gp_param = False
                try:
                    module_path = name.split('.')[:-1]
                    current_module = model
                    for part in module_path:
                        current_module = getattr(current_module, part)
                    
                    if isinstance(current_module, EnhancedGPLinear):
                        self.param_groups['gp_params'].append((name, param))
                        is_gp_param = True
                except:
                    pass
                
                if not is_gp_param:
                    if 'stages.' in name:
                        try:
                            stage_num = int(name.split('stages.')[1].split('.')[0])
                            if stage_num < num_stages // 3:
                                self.param_groups['early_stages'].append(param)
                            elif stage_num < 2 * num_stages // 3:
                                self.param_groups['mid_stages'].append(param)
                            else:
                                self.param_groups['late_stages'].append(param)
                        except:
                            self.param_groups['late_stages'].append(param)
                    else:
                        self.param_groups['late_stages'].append(param)
        
        # MAXIMUM LEARNING RATES
        self.optimizers = {}
        
        if self.param_groups['embedding']:
            self.optimizers['embedding'] = torch.optim.AdamW(
                self.param_groups['embedding'],
                lr=learning_rate * 0.5,  # 50% of base rate!
                betas=(0.9, 0.95),
                weight_decay=weight_decay * 0.1,
                eps=1e-8
            )
        
        if self.param_groups['lm_head']:
            self.optimizers['lm_head'] = torch.optim.AdamW(
                self.param_groups['lm_head'],
                lr=learning_rate * 0.5,  # 50% of base rate!
                betas=(0.9, 0.95),
                weight_decay=weight_decay * 0.1,
                eps=1e-8
            )
        
        if self.param_groups['transformation_ops']:
            self.optimizers['transformation_ops'] = torch.optim.AdamW(
                self.param_groups['transformation_ops'],
                lr=learning_rate * 0.1,  # 10% - still controlled
                betas=(0.9, 0.95),
                weight_decay=weight_decay * 0.1,
                eps=1e-8
            )
        
        if self.param_groups['layer_norm_params']:
            self.optimizers['layer_norm_params'] = torch.optim.AdamW(
                self.param_groups['layer_norm_params'],
                lr=learning_rate * 0.3,  # Higher
                betas=(0.9, 0.95),
                weight_decay=weight_decay * 0.1,
                eps=1e-8
            )
        
        if self.param_groups['early_stages']:
            self.optimizers['early_stages'] = torch.optim.AdamW(
                self.param_groups['early_stages'],
                lr=learning_rate * 0.8,  # Very high
                betas=(0.9, 0.95),
                weight_decay=weight_decay,
                eps=1e-8
            )
        
        if self.param_groups['mid_stages']:
            self.optimizers['mid_stages'] = torch.optim.AdamW(
                self.param_groups['mid_stages'],
                lr=learning_rate * 1.0,  # Full rate
                betas=(0.9, 0.95),
                weight_decay=weight_decay,
                eps=1e-8
            )
        
        if self.param_groups['late_stages']:
            self.optimizers['late_stages'] = torch.optim.AdamW(
                self.param_groups['late_stages'],
                lr=learning_rate * 1.2,  # Even higher!
                betas=(0.9, 0.95),
                weight_decay=weight_decay,
                eps=1e-8
            )
        
        if self.param_groups['heads']:
            self.optimizers['heads'] = torch.optim.AdamW(
                self.param_groups['heads'],
                lr=learning_rate * 0.5,  # High
                betas=(0.9, 0.95),
                weight_decay=weight_decay,
                eps=1e-8
            )
        
        if self.param_groups['vq_params']:
            self.optimizers['vq_params'] = torch.optim.AdamW(
                self.param_groups['vq_params'],
                lr=learning_rate * vq_lr_multiplier,
                betas=(0.9, 0.95),
                weight_decay=weight_decay,
                eps=1e-8
            )
        
        if self.param_groups['bias_params']:
            self.optimizers['bias_params'] = torch.optim.AdamW(
                self.param_groups['bias_params'],
                lr=learning_rate * 0.1,  # Much higher
                betas=(0.9, 0.95),
                weight_decay=weight_decay * 0.1,
                eps=1e-8
            )
        
        self.gp_learning_rate = learning_rate * gp_lr_multiplier
        
        print(f"MAXIMUM LEARNING RATE Optimizer:")
        for group_name, params in self.param_groups.items():
            if group_name == 'gp_params':
                count = len(params)
                total_params = sum(p.numel() for _, p in params)
            else:
                count = len(params)
                total_params = sum(p.numel() for p in params)
            
            if count > 0:
                group_lr = self._get_group_lr(group_name)
                print(f"  {group_name}: {count} params @ LR {group_lr:.2e}")
    
    def _get_group_lr(self, group_name):
        if group_name == 'embedding':
            return self.base_learning_rate * 0.5
        elif group_name == 'lm_head':
            return self.base_learning_rate * 0.5
        elif group_name == 'transformation_ops':
            return self.base_learning_rate * 0.1
        elif group_name == 'layer_norm_params':
            return self.base_learning_rate * 0.3
        elif group_name == 'early_stages':
            return self.base_learning_rate * 0.8
        elif group_name == 'mid_stages':
            return self.base_learning_rate * 1.0
        elif group_name == 'late_stages':
            return self.base_learning_rate * 1.2
        elif group_name == 'heads':
            return self.base_learning_rate * 0.5
        elif group_name == 'vq_params':
            return self.base_learning_rate * 0.2
        elif group_name == 'gp_params':
            return self.gp_learning_rate
        elif group_name == 'bias_params':
            return self.base_learning_rate * 0.1
        return self.base_learning_rate
    
    def zero_grad(self):
        for optimizer in self.optimizers.values():
            optimizer.zero_grad(set_to_none=True)
    
    def step(self):
        for optimizer in self.optimizers.values():
            optimizer.step()
        
        if self.weight_decay > 0:
            with torch.no_grad():
                for name, param in self.param_groups['gp_params']:
                    param.data.mul_(1 - self.gp_learning_rate * self.weight_decay)
    
    def state_dict(self):
        state = {}
        for name, optimizer in self.optimizers.items():
            state[f'optimizer_{name}'] = optimizer.state_dict()
        return state
    
    def load_state_dict(self, state_dict):
        for name, optimizer in self.optimizers.items():
            key = f'optimizer_{name}'
            if key in state_dict:
                optimizer.load_state_dict(state_dict[key])

class LearningRateWarmup:
    """MUCH FASTER warmup."""
    
    def __init__(self, optimizer, warmup_steps=500, total_steps=None):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.step_count = 0
        
        self.base_lrs = {}
        for name, opt in optimizer.optimizers.items():
            self.base_lrs[name] = opt.param_groups[0]['lr']
        
        self.base_lrs['gp_params'] = optimizer.gp_learning_rate
    
    def step(self):
        self.step_count += 1
        
        if self.step_count <= self.warmup_steps:
            # MUCH FASTER warmup - start at 50%
            warmup_factor = 0.5 + 0.5 * (self.step_count / self.warmup_steps)
        else:
            warmup_factor = 1.0
        
        for name, opt in self.optimizer.optimizers.items():
            new_lr = self.base_lrs[name] * warmup_factor
            for param_group in opt.param_groups:
                param_group['lr'] = new_lr
        
        self.optimizer.gp_learning_rate = self.base_lrs['gp_params'] * warmup_factor

def apply_minimal_gradient_clipping(model, gradient_monitor):
    """MINIMAL clipping - let gradients flow."""
    
    clipping_applied = {}
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_type = gradient_monitor._get_param_type(name)
            
            # Much higher clipping thresholds
            if 'wte' in name or 'lm_head' in name:
                clip_value = 50.0  # Very high
                torch.nn.utils.clip_grad_norm_([param], clip_value)
                clipping_applied[f"{name}_MINIMAL"] = clip_value
                
            elif 'transformation_ops' in name:
                clip_value = 10.0  # Much higher
                torch.nn.utils.clip_grad_norm_([param], clip_value)
                clipping_applied[f"{name}_TRANSFORM"] = clip_value
                
            elif ('ln' in name or 'norm' in name) and 'bias' in name:
                clip_value = 1.0  # Much higher
                torch.nn.utils.clip_grad_norm_([param], clip_value)
                clipping_applied[f"{name}_LN_BIAS"] = clip_value
                
            elif 'bias' in name:
                clip_value = 10.0  # Much higher
                torch.nn.utils.clip_grad_norm_([param], clip_value)
                clipping_applied[f"{name}_bias"] = clip_value
                
            else:
                # Much higher base clips
                base_clips = {
                    'embedding_params': 50.0,     # Very high
                    'lm_head_params': 50.0,       # Very high
                    'head_params': 20.0,          # High
                    'standard_params': 30.0,      # High
                    'gp_params': 20.0,            # High
                    'vq_params': 20.0,            # High
                    'transformation_ops': 10.0,   # Higher
                    'layer_norm_bias': 1.0        # Higher
                }
                clip_value = base_clips.get(param_type, 20.0)
                torch.nn.utils.clip_grad_norm_([param], clip_value)
                
                if param_type not in clipping_applied:
                    clipping_applied[param_type] = clip_value
    
    return clipping_applied

def hybrid_dmt_training_step(model, batch, scaler, grad_accum_steps, device, config, gradient_monitor, step_count):
    """AGGRESSIVE training with higher loss weights."""
    
    xb = batch[:, :config.block_size].to(device, non_blocking=True)
    yb = batch[:, 1:config.block_size+1].to(device, non_blocking=True)
    
    with torch.amp.autocast(device_type=device):
        try:
            logits, (ce_loss, q_loss, e_loss, conf_loss) = model(xb, yb)
            
            # MUCH MORE AGGRESSIVE loss balancing
            if step_count < 200:
                conf_weight = 0.01 * (step_count / 200.0)  # Faster, higher ramp
                q_weight = 0.1 * (step_count / 200.0)
            elif step_count < 1000:
                conf_weight = 0.01 + 0.04 * ((step_count - 200) / 800.0)
                q_weight = 0.1 + 0.4 * ((step_count - 200) / 800.0)
            else:
                conf_weight = 0.05  # Much higher
                q_weight = 0.5      # Much higher
            
            main_loss = (ce_loss + (conf_loss * conf_weight)) / grad_accum_steps
            vq_optimizer_loss = (q_loss * q_weight) / grad_accum_steps
            
            if not (torch.isfinite(main_loss) and torch.isfinite(vq_optimizer_loss)):
                print(f"Non-finite losses: main={main_loss.item()}, vq={vq_optimizer_loss.item()}")
                return 0.0, 0.0, 0.0, 0.0
            
            if main_loss.item() > 100.0:  # Much higher threshold
                print(f"Very high loss: {main_loss.item()}, skipping")
                return 0.0, 0.0, 0.0, 0.0
                
        except Exception as e:
            print(f"Forward pass error: {e}")
            return 0.0, 0.0, 0.0, 0.0
    
    try:
        scaler.scale(main_loss).backward(retain_graph=True)
        scaler.scale(vq_optimizer_loss).backward()
        
        return (main_loss.item() * grad_accum_steps, 
                ce_loss.item(), 
                conf_loss.item(), 
                q_loss.item())
    except Exception as e:
        print(f"Backward pass error: {e}")
        return 0.0, 0.0, 0.0, 0.0

# Keep utility functions unchanged but update the training parameters...

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
    LEARNING_RATE = 5e-4  # MUCH HIGHER base learning rate
    WARMUP_STEPS = 500    # MUCH FASTER warmup
    EVAL_EVERY = 20000
    CHECKPOINT_DIR = "checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    gradient_monitor = GradientMonitor()
    
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

    layerwise_optimizer = LayerwiseLearningRateOptimizer(
        model, 
        learning_rate=LEARNING_RATE,
        vq_lr_multiplier=0.2,   # Higher
        gp_lr_multiplier=0.3    # Much higher
    )
    
    lr_scheduler = LearningRateWarmup(layerwise_optimizer, warmup_steps=WARMUP_STEPS)
    
    scaler = torch.amp.GradScaler('cuda')
    
    if resume_from_checkpoint and 'optimizer_state_dict' in checkpoint:
        layerwise_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scaler_state_dict' in checkpoint: 
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        if 'lr_scheduler_step' in checkpoint:
            lr_scheduler.step_count = checkpoint['lr_scheduler_step']
    
    print(f"MAXIMUM LEARNING Hybrid DMT+GP Model: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")
    print(f"Effective batch size: {BATCH_SIZE * GRAD_ACCUM_STEPS}")
    print(f"FAST warmup: {WARMUP_STEPS} steps")
    
    if args.parquet_folder:
        if pd is None: 
            raise ImportError("Pandas is not installed. Please run 'pip install pandas pyarrow'.")
        
        parquet_files = sorted(glob.glob(os.path.join(args.parquet_folder, '*.parquet')))
        if not parquet_files: 
            raise ValueError(f"No .parquet files found in {args.parquet_folder}")
        
        print(f"Found {len(parquet_files)} parquet files for training.")
        print(f"Starting MAXIMUM LEARNING Hybrid DMT+GP Training...")
        
        dataset = TokenStreamDataset(parquet_files, tokenizer, args.text_column, config.block_size)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0, pin_memory=True)
        
        model.train()
        
        for micro_step, batch_chunk in enumerate(dataloader):
            loss, ce_loss, conf_loss, q_loss = hybrid_dmt_training_step(
                model, batch_chunk, scaler, GRAD_ACCUM_STEPS, device, config, gradient_monitor, global_step
            )
            
            if loss > 0:
                if (micro_step + 1) % GRAD_ACCUM_STEPS == 0:
                    try:
                        total_norm, max_grad = gradient_monitor.log_gradients(model, global_step)
                        
                        # MUCH HIGHER threshold - let it learn!
                        if total_norm > 50000.0:
                            print(f"🚨 EXTREME GRADIENT: {total_norm:.4f} - skipping")
                            layerwise_optimizer.zero_grad()
                            continue
                        
                        for optimizer in layerwise_optimizer.optimizers.values():
                            scaler.unscale_(optimizer)
                        
                        clipping_applied = apply_minimal_gradient_clipping(model, gradient_monitor)
                        
                        layerwise_optimizer.step()
                        lr_scheduler.step()
                        scaler.update()
                        layerwise_optimizer.zero_grad()
                        
                        global_step += 1
                        
                        if global_step % 100 == 0:
                            current_lr = lr_scheduler.optimizer.optimizers['late_stages'].param_groups[0]['lr']
                            warmup_factor = min(1.0, global_step / WARMUP_STEPS)
                            print(f"🚀 MAXIMUM LEARNING Step {global_step} | Loss {loss:.4f} (CE: {ce_loss:.4f}, Conf: {conf_loss:.4f}, Q: {q_loss:.4f}) | Grad: {total_norm:.4f} | LR: {current_lr:.2e} | Warmup: {warmup_factor:.3f}")
                        
                        if global_step > 0 and global_step % EVAL_EVERY == 0:
                            generate(model, tokenizer, device, prompt="The meaning of life is")
                            analysis_tokens_str = "This is a sample text for analyzing the model's internal state."
                            analysis_sample = torch.tensor(tokenizer.encode(analysis_tokens_str), dtype=torch.long)[:config.block_size]
                            display_bucket_contents(model, tokenizer, device, analysis_sample, stage_to_show=0, layer_to_show=0)
                            
                            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"maximum_learning_hybrid_dmt_gp_step_{global_step}.pth")
                            print(f"Saving checkpoint to {checkpoint_path}")
                            
                            torch.save({
                                'global_step': global_step, 
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': layerwise_optimizer.state_dict(),
                                'scaler_state_dict': scaler.state_dict(),
                                'lr_scheduler_step': lr_scheduler.step_count,
                                'config': config
                            }, checkpoint_path)
                    
                    except Exception as e:
                        print(f"Error in optimization step: {e}")
                        layerwise_optimizer.zero_grad()
            else:
                layerwise_optimizer.zero_grad()
    
    print("\n--- MAXIMUM LEARNING Hybrid DMT+GP Training Finished ---")
