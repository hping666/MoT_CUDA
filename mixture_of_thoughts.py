"""
Mixture of Thoughts (MoT) Framework - FIXED VERSION
Main implementation for orchestrating multiple LLMs with sparse top-k routing.

FIXES IN THIS VERSION:
1. InteractionLayer uses float32 internally for numerical stability
2. Better input normalization before projection
3. Smaller initialization scale for projection weights
4. More aggressive clamping of hidden states
5. LayerNorm before projection to stabilize inputs
6. Gradient scaling for mixed precision
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Tuple, Any
from dataclasses import dataclass, field
import math
from transformers import AutoModel, AutoTokenizer, PreTrainedModel
from sentence_transformers import SentenceTransformer


@dataclass
class MoTConfig:
    """Configuration for Mixture of Thoughts framework."""
    num_stacks: int = 4
    top_k: int = 3
    shared_dim: int = 768
    router_hidden_dim: int = 256
    router_temperature: float = 1.0
    enable_auxiliary_loss: bool = True
    interaction_heads: int = 8
    dropout_rate: float = 0.1
    sentence_encoder_model: str = 'microsoft/deberta-v3-large'
    compute_dtype: Optional[torch.dtype] = None
    # NEW: Stability settings
    use_stable_interaction: bool = True  # Use float32 in interaction layer
    interaction_scale: float = 0.05      # Reduced from 0.1
    clamp_hidden_states: float = 100.0   # Reduced from 1e4


class ExpertWrapper(nn.Module):
    """Wrapper for individual expert models with stack-based partitioning."""
    
    def __init__(self, model: PreTrainedModel, expert_id: int, num_stacks: int):
        super().__init__()
        self.model = model
        self.expert_id = expert_id
        self.num_stacks = num_stacks
        self.hidden_dim = model.config.hidden_size
        self.vocab_size = model.config.vocab_size
        
        self.model_type = self._detect_model_type(model)
        self.layers = self._get_layers(model)
        
        if self.layers is None:
            raise ValueError(f"Unsupported model architecture for expert {expert_id}: {type(model)}")
        
        self.num_layers = len(self.layers)
        self.stack_size = math.ceil(self.num_layers / num_stacks)
        self.compute_dtype = self._detect_compute_dtype(model)
        self.is_quantized = self._check_if_quantized(model)
        self.model_device = self._get_model_device()
        
        for param in self.model.parameters():
            param.requires_grad = False
        
        quant_str = "quantized" if self.is_quantized else f"{self.compute_dtype}"
        print(f"    Expert {expert_id}: {self.model_type}, {self.num_layers} layers, "
              f"hidden={self.hidden_dim}, vocab={self.vocab_size}, dtype={quant_str}, device={self.model_device}")
    
    def _get_model_device(self) -> torch.device:
        if hasattr(self.model, 'get_input_embeddings'):
            embed = self.model.get_input_embeddings()
            if embed is not None:
                for param in embed.parameters():
                    return param.device
        if self.layers is not None and len(self.layers) > 0:
            for param in self.layers[0].parameters():
                return param.device
        for param in self.model.parameters():
            return param.device
        return torch.device('cuda:0')
    
    def _detect_model_type(self, model) -> str:
        model_class = model.__class__.__name__.lower()
        config_class = model.config.__class__.__name__.lower()
        type_mapping = {
            'qwen': 'qwen', 'llama': 'llama', 'starcoder': 'starcoder2',
            'gpt2': 'gpt2', 'mistral': 'mistral', 'deepseek': 'deepseek',
            'phi': 'phi', 'gemma': 'gemma',
        }
        for key, value in type_mapping.items():
            if key in model_class or key in config_class:
                return value
        return 'generic'
    
    def _get_layers(self, model) -> Optional[nn.ModuleList]:
        paths_to_try = [
            ('model', 'layers'), ('transformer', 'h'), ('transformer', 'layers'),
            ('encoder', 'layer'), ('decoder', 'layers'), ('transformer', 'blocks'),
        ]
        for path in paths_to_try:
            obj = model
            try:
                for attr in path:
                    obj = getattr(obj, attr)
                if isinstance(obj, (nn.ModuleList, list)) and len(obj) > 0:
                    return obj
            except AttributeError:
                continue
        return None
    
    def _detect_compute_dtype(self, model) -> torch.dtype:
        for param in model.parameters():
            if param.dtype in [torch.float32, torch.float16, torch.bfloat16]:
                return param.dtype
        if hasattr(model.config, 'torch_dtype') and model.config.torch_dtype is not None:
            return model.config.torch_dtype
        return torch.float16
    
    def _check_if_quantized(self, model) -> bool:
        if hasattr(model, 'is_loaded_in_8bit') and model.is_loaded_in_8bit:
            return True
        if hasattr(model, 'is_loaded_in_4bit') and model.is_loaded_in_4bit:
            return True
        if hasattr(model.config, 'quantization_config'):
            return True
        for module in model.modules():
            module_class = module.__class__.__name__
            if 'Linear4bit' in module_class or 'Linear8bitLt' in module_class:
                return True
        return False
    
    def get_stack_boundaries(self) -> List[Tuple[int, int]]:
        boundaries = []
        for i in range(self.num_stacks):
            start = i * self.stack_size
            end = min((i + 1) * self.stack_size, self.num_layers)
            boundaries.append((start, end))
        return boundaries
    
    def get_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        input_ids = input_ids.to(self.model_device)
        if hasattr(self.model, 'get_input_embeddings'):
            embed_layer = self.model.get_input_embeddings()
            return embed_layer(input_ids)
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_tokens'):
            return self.model.model.embed_tokens(input_ids)
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'wte'):
            return self.model.transformer.wte(input_ids)
        else:
            raise ValueError(f"Cannot find embedding layer for expert {self.expert_id}")
    
    def forward_through_stack(
        self, 
        hidden_states: torch.Tensor,
        stack_idx: int,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        start, end = self.get_stack_boundaries()[stack_idx]
        batch_size, seq_len, hidden_dim = hidden_states.shape
        dtype = hidden_states.dtype
        device = self.model_device
        
        hidden_states = hidden_states.to(device)
        
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        else:
            position_ids = position_ids.to(device)
        
        attention_mask_2d = None
        if attention_mask is not None:
            attention_mask_2d = attention_mask.to(device)
        
        mask_value = -1e4
        causal_mask_4d = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=dtype) * mask_value,
            diagonal=1
        ).unsqueeze(0).unsqueeze(0)
        
        if attention_mask_2d is not None:
            padding_mask = attention_mask_2d[:, None, None, :].to(dtype)
            padding_mask = (1.0 - padding_mask) * mask_value
            combined_mask_4d = causal_mask_4d + padding_mask
        else:
            combined_mask_4d = causal_mask_4d
        
        position_embeddings = self._get_position_embeddings(hidden_states, position_ids, seq_len, device, dtype)
        
        for layer_idx in range(start, end):
            layer = self.layers[layer_idx]
            hidden_states = self._forward_single_layer(
                layer, hidden_states, attention_mask_2d, combined_mask_4d, 
                position_ids, position_embeddings, dtype
            )
            if torch.isnan(hidden_states).any():
                print(f"Warning: NaN detected after layer {layer_idx} in expert {self.expert_id}")
                hidden_states = torch.nan_to_num(hidden_states, nan=0.0)
        
        return hidden_states
    
    def _get_position_embeddings(
        self, hidden_states: torch.Tensor, position_ids: torch.Tensor, 
        seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        rotary_emb = None
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'rotary_emb'):
            rotary_emb = self.model.model.rotary_emb
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'rotary_emb'):
            rotary_emb = self.model.transformer.rotary_emb
        elif len(self.layers) > 0:
            first_layer = self.layers[0]
            if hasattr(first_layer, 'self_attn') and hasattr(first_layer.self_attn, 'rotary_emb'):
                rotary_emb = first_layer.self_attn.rotary_emb
        
        if rotary_emb is None:
            return None
        
        try:
            try:
                cos, sin = rotary_emb(hidden_states, position_ids)
                return (cos.to(dtype), sin.to(dtype))
            except TypeError:
                pass
            try:
                cos, sin = rotary_emb(hidden_states, seq_len=seq_len)
                return (cos.to(dtype), sin.to(dtype))
            except TypeError:
                pass
            try:
                cos, sin = rotary_emb(position_ids)
                return (cos.to(dtype), sin.to(dtype))
            except TypeError:
                pass
        except Exception as e:
            print(f"Warning: Failed to get position embeddings: {e}")
        return None
    
    def _forward_single_layer(
        self, layer: nn.Module, hidden_states: torch.Tensor,
        attention_mask_2d: Optional[torch.Tensor], attention_mask_4d: torch.Tensor,
        position_ids: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]],
        dtype: torch.dtype
    ) -> torch.Tensor:
        layer_class = layer.__class__.__name__
        call_attempts = []
        
        if position_embeddings is not None:
            call_attempts.extend([
                ('pos_emb + 4D mask full', {
                    'hidden_states': hidden_states, 'attention_mask': attention_mask_4d,
                    'position_ids': position_ids, 'position_embeddings': position_embeddings,
                    'use_cache': False, 'output_attentions': False,
                }),
                ('pos_emb + 4D mask', {
                    'hidden_states': hidden_states, 'attention_mask': attention_mask_4d,
                    'position_ids': position_ids, 'position_embeddings': position_embeddings,
                }),
                ('pos_emb only', {
                    'hidden_states': hidden_states, 'position_ids': position_ids,
                    'position_embeddings': position_embeddings,
                }),
            ])
        
        call_attempts.extend([
            ('4D mask + pos_ids', {
                'hidden_states': hidden_states, 'attention_mask': attention_mask_4d,
                'position_ids': position_ids, 'use_cache': False, 'output_attentions': False,
            }),
            ('4D mask minimal', {
                'hidden_states': hidden_states, 'attention_mask': attention_mask_4d,
                'position_ids': position_ids,
            }),
            ('no mask', {'hidden_states': hidden_states, 'position_ids': position_ids}),
            ('hidden only', {'hidden_states': hidden_states}),
        ])
        
        errors_log = []
        for attempt_name, attempt_kwargs in call_attempts:
            try:
                outputs = layer(**attempt_kwargs)
                if outputs is None:
                    errors_log.append(f"{attempt_name}: outputs is None")
                    continue
                if isinstance(outputs, tuple):
                    if len(outputs) > 0 and outputs[0] is not None:
                        return outputs[0]
                    else:
                        errors_log.append(f"{attempt_name}: tuple[0] is None")
                        continue
                if hasattr(outputs, 'last_hidden_state') and outputs.last_hidden_state is not None:
                    return outputs.last_hidden_state
                if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                    hs = outputs.hidden_states
                    if isinstance(hs, tuple) and len(hs) > 0:
                        return hs[-1]
                    return hs
                if isinstance(outputs, torch.Tensor):
                    return outputs
                errors_log.append(f"{attempt_name}: unknown type {type(outputs)}")
            except TypeError as e:
                errors_log.append(f"{attempt_name}: TypeError - {str(e)[:80]}")
            except torch.cuda.OutOfMemoryError as e:
                torch.cuda.empty_cache()
                raise RuntimeError(f"CUDA OOM in {layer_class}. Reduce max_length or batch_size.")
            except RuntimeError as e:
                err_str = str(e)
                if "out of memory" in err_str.lower():
                    torch.cuda.empty_cache()
                    raise RuntimeError(f"CUDA OOM in {layer_class}.")
                if "device" in err_str.lower():
                    raise RuntimeError(f"Device mismatch. Use --single_gpu.")
                errors_log.append(f"{attempt_name}: RuntimeError - {err_str[:80]}")
            except Exception as e:
                errors_log.append(f"{attempt_name}: {type(e).__name__} - {str(e)[:80]}")
        
        pe_status = "available" if position_embeddings is not None else "NOT AVAILABLE"
        error_msg = f"Failed to forward through {layer_class}. position_embeddings: {pe_status}\nAttempts:\n" + "\n".join(errors_log)
        raise RuntimeError(error_msg)


class SparseRouter(nn.Module):
    """Sparse top-k router for expert selection. Always uses float32."""
    
    def __init__(self, input_dim: int, num_experts: int, config: MoTConfig):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(config.top_k, num_experts)
        self.temperature = config.router_temperature
        
        self.router_mlp = nn.Sequential(
            nn.Linear(input_dim, config.router_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.router_hidden_dim, config.router_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.router_hidden_dim, num_experts)
        )
        
        for layer in self.router_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=0.02)
                nn.init.zeros_(layer.bias)
    
    def forward(self, prompt_embedding: torch.Tensor, return_scores: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prompt_float = prompt_embedding.float()
        scores = self.router_mlp(prompt_float)
        scores = scores / self.temperature
        
        top_k_scores, top_k_indices = torch.topk(scores, self.top_k, dim=-1)
        expert_weights = F.softmax(top_k_scores, dim=-1)
        primary_expert = top_k_indices[:, 0]
        
        if return_scores:
            return top_k_indices, expert_weights, primary_expert, scores
        return top_k_indices, expert_weights, primary_expert
    
    def compute_load_balancing_loss(self, expert_indices: torch.Tensor) -> torch.Tensor:
        batch_size = expert_indices.size(0)
        expert_counts = torch.zeros(self.num_experts, device=expert_indices.device, dtype=torch.float32)
        for i in range(self.num_experts):
            expert_counts[i] = (expert_indices == i).float().sum()
        expert_freq = expert_counts / (batch_size * self.top_k + 1e-8)
        mean_freq = expert_freq.mean()
        std_freq = expert_freq.std()
        return (std_freq / (mean_freq + 1e-8)) ** 2
    
    def compute_entropy_loss(self, scores: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(scores / self.temperature, dim=-1)
        log_probs = F.log_softmax(scores / self.temperature, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        return -entropy


class InteractionLayer(nn.Module):
    """
    Cross-attention based interaction layer for expert communication.
    
    FIXED: Now uses float32 internally for numerical stability with quantized models.
    """
    
    def __init__(self, expert_dims: List[int], config: MoTConfig):
        super().__init__()
        self.shared_dim = config.shared_dim
        self.num_heads = config.interaction_heads
        self.head_dim = self.shared_dim // self.num_heads
        self.num_experts = len(expert_dims)
        self.interaction_scale = config.interaction_scale
        self.clamp_value = config.clamp_hidden_states
        
        # Input LayerNorms for each expert (stabilizes input before projection)
        self.input_norms = nn.ModuleList([
            nn.LayerNorm(dim) for dim in expert_dims
        ])
        
        # Projection layers
        self.expert_projections = nn.ModuleList([
            nn.Linear(dim, self.shared_dim) for dim in expert_dims
        ])
        
        self.q_proj = nn.Linear(self.shared_dim, self.shared_dim)
        self.k_proj = nn.Linear(self.shared_dim, self.shared_dim)
        self.v_proj = nn.Linear(self.shared_dim, self.shared_dim)
        self.out_proj = nn.Linear(self.shared_dim, self.shared_dim)
        
        self.back_projections = nn.ModuleList([
            nn.Linear(self.shared_dim, dim) for dim in expert_dims
        ])
        
        self.residual_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.shared_dim, self.shared_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout_rate),
                nn.Linear(self.shared_dim, self.shared_dim)
            ) for _ in expert_dims
        ])
        
        self.dropout = nn.Dropout(config.dropout_rate)
        self.layer_norm = nn.LayerNorm(self.shared_dim)
        
        # Output LayerNorms for stability
        self.output_norms = nn.ModuleList([
            nn.LayerNorm(dim) for dim in expert_dims
        ])
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with smaller values for stability."""
        # Use smaller initialization for projection layers
        init_scale = 0.02
        
        for i, proj in enumerate(self.expert_projections):
            nn.init.normal_(proj.weight, mean=0.0, std=init_scale)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)
        
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.normal_(proj.weight, mean=0.0, std=init_scale)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)
        
        # Even smaller for back projection
        for back_proj in self.back_projections:
            nn.init.normal_(back_proj.weight, mean=0.0, std=init_scale * 0.1)
            if back_proj.bias is not None:
                nn.init.zeros_(back_proj.bias)
        
        for res_seq in self.residual_projections:
            for layer in res_seq:
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, mean=0.0, std=init_scale * 0.1)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
    
    def _align_sequence_length(self, hidden_states: torch.Tensor, target_length: int) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        if seq_len == target_length:
            return hidden_states
        hidden_t = hidden_states.float().transpose(1, 2)
        aligned = F.interpolate(hidden_t, size=target_length, mode='linear', align_corners=True)
        return aligned.transpose(1, 2)
    
    def forward(
        self,
        expert_hidden_states: List[torch.Tensor],
        expert_attention_masks: List[torch.Tensor],
        active_experts: torch.Tensor,
        primary_expert: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Perform inter-expert communication with improved numerical stability.
        All computations done in float32 internally.
        """
        primary_idx = primary_expert[0].item()
        primary_hidden = expert_hidden_states[primary_idx]
        
        if torch.isnan(primary_hidden).any():
            print(f"Warning: NaN in InteractionLayer input, returning original states")
            return expert_hidden_states, expert_attention_masks
        
        batch_size = primary_hidden.size(0)
        target_seq_len = primary_hidden.size(1)
        
        # Get device from first projection layer
        layer_device = self.expert_projections[0].weight.device
        
        aligned_states = []
        aligned_masks = []
        active_indices = []
        original_dtypes = {}
        original_devices = {}
        
        for idx in active_experts[0]:
            idx_val = idx.item()
            if expert_hidden_states[idx_val] is None:
                continue
            
            hidden = expert_hidden_states[idx_val]
            mask = expert_attention_masks[idx_val]
            
            if torch.isnan(hidden).any():
                print(f"Warning: NaN in expert {idx_val} hidden states, skipping interaction")
                continue
            
            original_dtypes[idx_val] = hidden.dtype
            original_devices[idx_val] = hidden.device
            
            # === KEY FIX: Use float32 for all interaction computations ===
            hidden_f32 = hidden.to(device=layer_device, dtype=torch.float32)
            
            # Clamp to reasonable range BEFORE any operations
            hidden_f32 = torch.clamp(hidden_f32, min=-self.clamp_value, max=self.clamp_value)
            
            # Apply input LayerNorm for stability
            hidden_normed = self.input_norms[idx_val].float()(hidden_f32)
            
            # Check for NaN after normalization
            if torch.isnan(hidden_normed).any():
                print(f"Warning: NaN after LayerNorm for expert {idx_val}, using clamped input")
                hidden_normed = hidden_f32
            
            # Project to shared space (in float32)
            proj_weight = self.expert_projections[idx_val].weight.float()
            proj_bias = self.expert_projections[idx_val].bias
            if proj_bias is not None:
                proj_bias = proj_bias.float()
            
            proj = F.linear(hidden_normed, proj_weight, proj_bias)
            
            # Check for NaN after projection
            if torch.isnan(proj).any():
                print(f"Warning: NaN after projection for expert {idx_val}, skipping")
                continue
            
            # Clamp projection output
            proj = torch.clamp(proj, min=-self.clamp_value, max=self.clamp_value)
            
            aligned = self._align_sequence_length(proj, target_seq_len)
            aligned_states.append(aligned)
            
            mask_on_device = mask.to(layer_device)
            if mask_on_device.size(1) != target_seq_len:
                mask_float = mask_on_device.float().unsqueeze(1)
                aligned_mask = F.interpolate(mask_float, size=target_seq_len, mode='nearest')
                aligned_mask = (aligned_mask.squeeze(1) > 0.5).long()
            else:
                aligned_mask = mask_on_device
            aligned_masks.append(aligned_mask)
            active_indices.append(idx_val)
        
        if not aligned_states:
            return expert_hidden_states, expert_attention_masks
        
        # Cross-attention computation (all in float32)
        primary_local_idx = active_indices.index(primary_idx) if primary_idx in active_indices else 0
        primary_proj = aligned_states[primary_local_idx]
        
        # Q, K, V projections in float32
        Q = F.linear(primary_proj, self.q_proj.weight.float(), 
                     self.q_proj.bias.float() if self.q_proj.bias is not None else None)
        Q = Q.view(batch_size, target_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        all_states = torch.cat(aligned_states, dim=1)
        total_kv_len = all_states.size(1)
        
        K = F.linear(all_states, self.k_proj.weight.float(),
                     self.k_proj.bias.float() if self.k_proj.bias is not None else None)
        K = K.view(batch_size, total_kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        V = F.linear(all_states, self.v_proj.weight.float(),
                     self.v_proj.bias.float() if self.v_proj.bias is not None else None)
        V = V.view(batch_size, total_kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores with stability
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = torch.clamp(scores, min=-50, max=50)
        
        combined_mask = torch.cat(aligned_masks, dim=1)
        expanded_mask = combined_mask.unsqueeze(1).unsqueeze(2).expand(
            batch_size, self.num_heads, target_seq_len, total_kv_len
        ).float()
        scores = scores.masked_fill(expanded_mask == 0, -1e4)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, target_seq_len, self.shared_dim
        )
        
        out_proj = F.linear(attn_output, self.out_proj.weight.float(),
                           self.out_proj.bias.float() if self.out_proj.bias is not None else None)
        
        if torch.isnan(out_proj).any():
            print(f"Warning: NaN in attn_output, returning original states")
            return expert_hidden_states, expert_attention_masks
        
        # Layer norm (in float32)
        layer_norm_weight = self.layer_norm.weight.float()
        layer_norm_bias = self.layer_norm.bias.float() if self.layer_norm.bias is not None else None
        attn_output = F.layer_norm(out_proj + primary_proj, [self.shared_dim], 
                                   layer_norm_weight, layer_norm_bias)
        
        # Update each expert's hidden states
        updated_hidden_states = list(expert_hidden_states)
        updated_attention_masks = list(expert_attention_masks)
        
        for i, idx_val in enumerate(active_indices):
            original_seq_len = expert_hidden_states[idx_val].size(1)
            original_dtype = original_dtypes[idx_val]
            original_device = original_devices[idx_val]
            
            if idx_val == primary_idx:
                update_source = attn_output
            else:
                proj = aligned_states[i]
                # Residual projection in float32
                residual = proj
                for layer in self.residual_projections[idx_val]:
                    if isinstance(layer, nn.Linear):
                        residual = F.linear(residual, layer.weight.float(),
                                          layer.bias.float() if layer.bias is not None else None)
                    elif isinstance(layer, nn.ReLU):
                        residual = F.relu(residual)
                    elif isinstance(layer, nn.Dropout):
                        residual = self.dropout(residual)
                update_source = residual
            
            # Align back to original sequence length
            if original_seq_len != target_seq_len:
                update_t = F.interpolate(
                    update_source.transpose(1, 2),
                    size=original_seq_len,
                    mode='linear',
                    align_corners=True
                ).transpose(1, 2)
            else:
                update_t = update_source
            
            # Back projection in float32
            back_proj = F.linear(update_t, self.back_projections[idx_val].weight.float(),
                                self.back_projections[idx_val].bias.float() 
                                if self.back_projections[idx_val].bias is not None else None)
            
            if torch.isnan(back_proj).any():
                print(f"Warning: NaN in back_proj for expert {idx_val}, skipping update")
                continue
            
            # Clamp and scale down the update
            back_proj = torch.clamp(back_proj, min=-self.clamp_value, max=self.clamp_value)
            
            # Apply output normalization
            back_proj_normed = F.layer_norm(
                back_proj, [expert_hidden_states[idx_val].size(-1)],
                self.output_norms[idx_val].weight.float(),
                self.output_norms[idx_val].bias.float() if self.output_norms[idx_val].bias is not None else None
            )
            
            # Convert back to original dtype and device
            back_proj_original = back_proj_normed.to(device=original_device, dtype=original_dtype)
            
            # Small scale factor for stability
            updated_hidden_states[idx_val] = expert_hidden_states[idx_val] + self.interaction_scale * back_proj_original
        
        return updated_hidden_states, updated_attention_masks


class MixtureOfThoughts(nn.Module):
    """Main Mixture of Thoughts framework with improved stability."""
    
    def __init__(
        self,
        expert_models: List[PreTrainedModel],
        tokenizers: List[AutoTokenizer],
        config: MoTConfig
    ):
        super().__init__()
        self.config = config
        self.num_experts = len(expert_models)
        
        print(f"  Initializing {self.num_experts} expert wrappers...")
        
        self.experts = nn.ModuleList([
            ExpertWrapper(model, i, config.num_stacks)
            for i, model in enumerate(expert_models)
        ])
        self.tokenizers = tokenizers
        
        for tokenizer in self.tokenizers:
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
        
        self.expert_dims = [expert.hidden_dim for expert in self.experts]
        self.vocab_sizes = [expert.vocab_size for expert in self.experts]
        
        if config.compute_dtype is None:
            self.compute_dtype = self._detect_compute_dtype()
        else:
            self.compute_dtype = config.compute_dtype
        
        print(f"  Expert vocab sizes: {self.vocab_sizes}")
        print(f"  Expert hidden dims: {self.expert_dims}")
        print(f"  Compute dtype: {self.compute_dtype}")
        
        self._init_sentence_encoder(config)
        
        sentence_encoder_dim = self.sentence_encoder.get_sentence_embedding_dimension()
        
        # Router always in float32
        self.router = SparseRouter(sentence_encoder_dim, self.num_experts, config)
        
        # Interaction layers - will use float32 internally
        self.interaction_layers = nn.ModuleList([
            InteractionLayer(self.expert_dims, config)
            for _ in range(config.num_stacks)
        ])
        
        if config.enable_auxiliary_loss:
            self.auxiliary_heads = nn.ModuleList([
                nn.Linear(dim, vocab_size)
                for dim, vocab_size in zip(self.expert_dims, self.vocab_sizes)
            ])
        
        self._setup_trainable_layers_dtype()
    
    def _detect_compute_dtype(self) -> torch.dtype:
        any_quantized = any(expert.is_quantized for expert in self.experts)
        if any_quantized:
            return torch.float16
        return self.experts[0].compute_dtype
    
    def _init_sentence_encoder(self, config: MoTConfig):
        if 'deberta-v3' in config.sentence_encoder_model.lower():
            from sentence_transformers import models
            word_embedding_model = models.Transformer(
                config.sentence_encoder_model, max_seq_length=512
            )
            pooling_model = models.Pooling(
                word_embedding_model.get_word_embedding_dimension(),
                pooling_mode_mean_tokens=True
            )
            self.sentence_encoder = SentenceTransformer(
                modules=[word_embedding_model, pooling_model]
            )
            print(f"  Initialized DeBERTa-v3 sentence encoder")
        else:
            self.sentence_encoder = SentenceTransformer(config.sentence_encoder_model)
            print(f"  Loaded sentence encoder: {config.sentence_encoder_model}")
    
    def _setup_trainable_layers_dtype(self):
        """Setup dtype for trainable layers."""
        # Interaction layers keep float32 parameters but can receive float16 inputs
        # The forward pass handles conversion internally
        for layer in self.interaction_layers:
            layer.to(torch.float32)  # Keep in float32 for stability
        
        if hasattr(self, 'auxiliary_heads'):
            for head in self.auxiliary_heads:
                head.to(self.compute_dtype)
        
        self.router.to(torch.float32)
        
        print(f"  Interaction layers dtype: float32 (internal computation)")
        print(f"  Router dtype: float32 (for stability)")
    
    def _get_device(self) -> torch.device:
        return next(self.router.parameters()).device
    
    def encode_prompt(self, texts: List[str]) -> torch.Tensor:
        device = self._get_device()
        prompt_repr = self.sentence_encoder.encode(
            texts, convert_to_tensor=True, device=str(device)
        )
        if prompt_repr.device != device:
            prompt_repr = prompt_repr.to(device)
        return prompt_repr.clone()
    
    def _tokenize_for_expert(
        self, texts: List[str], expert_idx: int, max_length: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tokenizer = self.tokenizers[expert_idx]
        encoding = tokenizer(
            texts, max_length=max_length, padding='max_length',
            truncation=True, return_tensors='pt'
        )
        return encoding['input_ids'].to(device), encoding['attention_mask'].to(device)
    
    def _create_expert_labels(
        self, raw_texts: List[str], expert_idx: int,
        expert_input_ids: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        tokenizer = self.tokenizers[expert_idx]
        labels = expert_input_ids.clone()
        
        for i, text in enumerate(raw_texts):
            cuda_marker = "### CUDA:\n"
            marker_pos = text.find(cuda_marker)
            if marker_pos != -1:
                prompt_text = text[:marker_pos + len(cuda_marker)]
                prompt_encoding = tokenizer(prompt_text, add_special_tokens=True, return_tensors='pt')
                prompt_len = prompt_encoding['input_ids'].shape[1]
                labels[i, :prompt_len] = -100
        
        labels[expert_input_ids == tokenizer.pad_token_id] = -100
        return labels
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        raw_texts: Optional[List[str]] = None,
        prompt_lengths: Optional[List[int]] = None,
        return_dict: bool = True
    ) -> Dict[str, Any]:
        device = self._get_device()
        
        if raw_texts is None:
            if input_ids is None:
                raise ValueError("Either input_ids or raw_texts must be provided")
            raw_texts = []
            for i in range(input_ids.size(0)):
                if attention_mask is not None:
                    valid_len = int(attention_mask[i].sum().item())
                    valid_ids = input_ids[i][:valid_len]
                else:
                    valid_ids = input_ids[i]
                text = self.tokenizers[0].decode(valid_ids.cpu(), skip_special_tokens=False)
                raw_texts.append(text)
        
        batch_size = len(raw_texts)
        max_length = input_ids.size(1) if input_ids is not None else 2048
        
        prompt_repr = self.encode_prompt(raw_texts)
        
        active_experts, expert_weights, primary_expert, all_scores = self.router(
            prompt_repr, return_scores=True
        )
        
        expert_hidden_states = [None] * self.num_experts
        expert_attention_masks = [None] * self.num_experts
        expert_input_ids_list = [None] * self.num_experts
        expert_labels_list = [None] * self.num_experts
        
        for expert_idx in active_experts[0]:
            expert_idx = expert_idx.item()
            
            exp_input_ids, exp_attention_mask = self._tokenize_for_expert(
                raw_texts, expert_idx, max_length, device
            )
            
            expert_input_ids_list[expert_idx] = exp_input_ids
            expert_attention_masks[expert_idx] = exp_attention_mask
            
            expert = self.experts[expert_idx]
            hidden_states = expert.get_embeddings(exp_input_ids)
            expert_hidden_states[expert_idx] = hidden_states
            
            if labels is not None:
                expert_labels_list[expert_idx] = self._create_expert_labels(
                    raw_texts, expert_idx, exp_input_ids, device
                )
        
        for stack_idx in range(self.config.num_stacks):
            for expert_idx in active_experts[0]:
                expert_idx = expert_idx.item()
                expert = self.experts[expert_idx]
                
                hidden = expert.forward_through_stack(
                    expert_hidden_states[expert_idx],
                    stack_idx,
                    attention_mask=expert_attention_masks[expert_idx]
                )
                expert_hidden_states[expert_idx] = hidden
            
            if stack_idx < self.config.num_stacks - 1:
                expert_hidden_states, expert_attention_masks = self.interaction_layers[stack_idx](
                    expert_hidden_states,
                    expert_attention_masks,
                    active_experts,
                    primary_expert
                )
        
        primary_idx = primary_expert[0].item()
        primary_expert_model = self.experts[primary_idx].model
        primary_hidden = expert_hidden_states[primary_idx]
        
        if hasattr(self.experts[primary_idx], 'model'):
            model = self.experts[primary_idx].model
            norm_layer = None
            if hasattr(model, 'model') and hasattr(model.model, 'norm'):
                norm_layer = model.model.norm
            elif hasattr(model, 'transformer') and hasattr(model.transformer, 'ln_f'):
                norm_layer = model.transformer.ln_f
            elif hasattr(model, 'model') and hasattr(model.model, 'final_layernorm'):
                norm_layer = model.model.final_layernorm
            
            if norm_layer is not None:
                norm_device = next(norm_layer.parameters()).device
                primary_hidden = norm_layer(primary_hidden.to(norm_device))
        
        if torch.isnan(primary_hidden).any():
            print(f"Warning: NaN in primary_hidden before lm_head, replacing with zeros")
            primary_hidden = torch.nan_to_num(primary_hidden, nan=0.0)
        
        if hasattr(primary_expert_model, 'lm_head'):
            lm_head = primary_expert_model.lm_head
            lm_head_device = self.experts[primary_idx].model_device
            primary_hidden_for_lm = primary_hidden.to(lm_head_device)
            logits = lm_head(primary_hidden_for_lm)
        elif hasattr(primary_expert_model, 'cls'):
            cls_head = primary_expert_model.cls
            cls_device = self.experts[primary_idx].model_device
            primary_hidden_for_cls = primary_hidden.to(cls_device)
            logits = cls_head(primary_hidden_for_cls)
        else:
            raise ValueError("Primary expert doesn't have output head")
        
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f"Warning: NaN/Inf in logits, clamping")
            logits = torch.nan_to_num(logits, nan=0.0, posinf=100.0, neginf=-100.0)
        
        total_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            
            primary_labels = expert_labels_list[primary_idx]
            if primary_labels is None:
                primary_labels = self._create_expert_labels(
                    raw_texts, primary_idx, expert_input_ids_list[primary_idx], device
                )
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = primary_labels[..., 1:].contiguous()
            shift_labels = shift_labels.to(shift_logits.device)
            
            shift_logits_clamped = torch.clamp(shift_logits, min=-100, max=100)
            
            lm_loss = loss_fct(
                shift_logits_clamped.view(-1, shift_logits_clamped.size(-1)).float(),
                shift_labels.view(-1)
            )
            
            if torch.isnan(lm_loss) or torch.isinf(lm_loss):
                print(f"Warning: lm_loss is {lm_loss.item()}, replacing with 0")
                lm_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
            
            entropy_loss = self.router.compute_entropy_loss(all_scores)
            balance_loss = self.router.compute_load_balancing_loss(active_experts)
            
            if torch.isnan(entropy_loss) or torch.isinf(entropy_loss):
                entropy_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
            if torch.isnan(balance_loss) or torch.isinf(balance_loss):
                balance_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
            
            aux_loss = torch.tensor(0.0, device=lm_loss.device, dtype=torch.float32)
            
            lm_loss = lm_loss.to(device)
            entropy_loss = entropy_loss.to(device)
            balance_loss = balance_loss.to(device)
            aux_loss = aux_loss.to(device)
            
            total_loss = lm_loss + 0.01 * entropy_loss + 0.01 * balance_loss
            
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"Warning: total_loss is NaN/Inf")
                total_loss = torch.tensor(0.0, device=device, dtype=torch.float32, requires_grad=True)
        
        if return_dict:
            return {
                'loss': total_loss,
                'logits': logits,
                'active_experts': active_experts,
                'expert_weights': expert_weights,
                'primary_expert': primary_expert,
                'router_scores': all_scores,
                'hidden_states': expert_hidden_states
            }
        
        return (total_loss, logits)
    
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        raw_text: Optional[str] = None,
        max_length: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs
    ) -> torch.Tensor:
        device = self._get_device()
        
        if raw_text is None:
            if input_ids is None:
                raise ValueError("Either input_ids or raw_text must be provided")
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            valid_len = int(attention_mask[0].sum().item())
            valid_ids = input_ids[0][:valid_len]
            raw_text = self.tokenizers[0].decode(valid_ids.cpu(), skip_special_tokens=False)
        
        prompt_repr = self.encode_prompt([raw_text])
        _, _, primary_expert = self.router(prompt_repr)
        primary_idx = primary_expert[0].item()
        
        primary_tokenizer = self.tokenizers[primary_idx]
        primary_model = self.experts[primary_idx].model
        
        encoding = primary_tokenizer(
            raw_text, return_tensors='pt', truncation=True,
            max_length=max_length // 2 if max_length else 1024
        )
        gen_input_ids = encoding['input_ids'].to(device)
        gen_attention_mask = encoding['attention_mask'].to(device)
        
        if max_new_tokens is not None and max_length is None:
            max_length = gen_input_ids.shape[1] + max_new_tokens
        elif max_length is None:
            max_length = 2048
        
        with torch.no_grad():
            gen_kwargs = {k: v for k, v in kwargs.items() 
                         if k not in ['do_sample', 'temperature', 'top_p', 'max_length', 'max_new_tokens']}
            
            generated = primary_model.generate(
                input_ids=gen_input_ids,
                attention_mask=gen_attention_mask,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=primary_tokenizer.pad_token_id,
                eos_token_id=primary_tokenizer.eos_token_id,
                **gen_kwargs
            )
        
        return generated
    
    def translate(
        self,
        source_text: str,
        prompt_template: str = "### Translate C++ to CUDA:\n{source}\n### CUDA:\n",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        **kwargs
    ) -> Tuple[str, int]:
        prompt = prompt_template.format(source=source_text)
        
        prompt_repr = self.encode_prompt([prompt])
        _, _, primary_expert = self.router(prompt_repr)
        primary_idx = primary_expert[0].item()
        
        tokenizer = self.tokenizers[primary_idx]
        
        output_ids = self.generate(
            raw_text=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs
        )
        
        prompt_encoding = tokenizer(prompt, return_tensors='pt')
        prompt_len = prompt_encoding['input_ids'].shape[1]
        
        generated_ids = output_ids[0, prompt_len:]
        generated_code = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        for stop_marker in ['\n\n\n', '###', '<|endoftext|>', '<|end|>', '<|im_end|>']:
            if stop_marker in generated_code:
                generated_code = generated_code.split(stop_marker)[0]
        
        return generated_code.strip(), primary_idx
    
    def get_routing_info(self, text: str) -> Dict[str, Any]:
        with torch.no_grad():
            prompt_repr = self.encode_prompt([text])
            active_experts, expert_weights, primary_expert, scores = self.router(
                prompt_repr, return_scores=True
            )
            all_probs = F.softmax(scores, dim=-1)
            
            return {
                'active_experts': active_experts.cpu().tolist(),
                'expert_weights': expert_weights.cpu().tolist(),
                'primary_expert': primary_expert.cpu().item(),
                'all_expert_probs': all_probs.cpu().tolist(),
                'router_scores': scores.cpu().tolist()
            }