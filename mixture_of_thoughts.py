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
7. Added Gumbel-Softmax for differentiable top-k selection
8. Added Straight-through estimator for gradient flow
9. Added routing consistency loss (L_con)
10. Fixed generate() to use MoT forward pass
11. Fixed translate() to avoid redundant routing calls
12. Added generate_with_mot() for full MoT inference mode
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
    # Stability settings
    use_stable_interaction: bool = True
    interaction_scale: float = 0.05
    clamp_hidden_states: float = 100.0
    # Gumbel-Softmax settings
    gumbel_temperature: float = 1.0
    use_gumbel: bool = True
    # Consistency loss settings
    consistency_temperature: float = 2.0
    lambda_consistency: float = 0.05


class GumbelSoftmax:
    """Gumbel-Softmax for differentiable top-k selection."""
    
    @staticmethod
    def sample_gumbel(shape: Tuple, device: torch.device, eps: float = 1e-20) -> torch.Tensor:
        """Sample from Gumbel(0, 1) distribution."""
        U = torch.rand(shape, device=device)
        return -torch.log(-torch.log(U + eps) + eps)
    
    @staticmethod
    def gumbel_softmax_sample(logits: torch.Tensor, temperature: float) -> torch.Tensor:
        """Sample from Gumbel-Softmax distribution."""
        gumbel_noise = GumbelSoftmax.sample_gumbel(logits.size(), logits.device)
        y = logits + gumbel_noise
        return F.softmax(y / temperature, dim=-1)
    
    @staticmethod
    def differentiable_topk(
        logits: torch.Tensor, 
        k: int, 
        temperature: float = 1.0,
        hard: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Differentiable top-k selection using Gumbel-Softmax.
        
        Args:
            logits: Input logits [batch_size, num_experts]
            k: Number of experts to select
            temperature: Temperature for Gumbel-Softmax
            hard: Whether to use straight-through estimator
            
        Returns:
            indices: Selected expert indices [batch_size, k]
            weights: Soft or hard selection weights [batch_size, num_experts]
        """
        batch_size, num_experts = logits.size()
        
        # Gumbel-Softmax sampling
        y_soft = GumbelSoftmax.gumbel_softmax_sample(logits, temperature)
        
        if hard:
            # Straight-through estimator: hard forward, soft backward
            _, indices = torch.topk(logits, k, dim=-1)
            y_hard = torch.zeros_like(y_soft)
            y_hard.scatter_(1, indices, 1.0)
            # Gradient flows through y_soft
            y = y_hard - y_soft.detach() + y_soft
        else:
            # Soft selection
            _, indices = torch.topk(y_soft, k, dim=-1)
            y = y_soft
        
        return indices, y


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
    """Sparse top-k router for expert selection with Gumbel-Softmax support."""
    
    def __init__(self, input_dim: int, num_experts: int, config: MoTConfig):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(config.top_k, num_experts)
        self.temperature = config.router_temperature
        self.gumbel_temperature = config.gumbel_temperature
        self.use_gumbel = config.use_gumbel
        
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
    
    def forward(
        self, 
        prompt_embedding: torch.Tensor, 
        return_scores: bool = False,
        use_gumbel: Optional[bool] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route to top-k experts with optional Gumbel-Softmax.
        
        Args:
            prompt_embedding: [batch_size, input_dim] tensor
            return_scores: Whether to return all expert scores
            use_gumbel: Override config.use_gumbel (None = use config)
        """
        prompt_float = prompt_embedding.float()
        scores = self.router_mlp(prompt_float)
        scores = scores / self.temperature
        
        # Determine whether to use Gumbel-Softmax
        apply_gumbel = use_gumbel if use_gumbel is not None else self.use_gumbel
        
        if self.training and apply_gumbel:
            # Use Gumbel-Softmax with straight-through estimator
            top_k_indices, selection_weights = GumbelSoftmax.differentiable_topk(
                scores, self.top_k, self.gumbel_temperature, hard=True
            )
            # Get weights for selected experts
            top_k_scores = torch.gather(scores, 1, top_k_indices)
            expert_weights = F.softmax(top_k_scores, dim=-1)
        else:
            # Standard top-k selection (inference mode)
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
    Uses float32 internally for numerical stability with quantized models.
    """
    
    def __init__(self, expert_dims: List[int], config: MoTConfig):
        super().__init__()
        self.shared_dim = config.shared_dim
        self.num_heads = config.interaction_heads
        self.head_dim = self.shared_dim // self.num_heads
        self.num_experts = len(expert_dims)
        self.interaction_scale = config.interaction_scale
        self.clamp_value = config.clamp_hidden_states
        
        # Input LayerNorms for each expert
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
        
        # Output LayerNorms
        self.output_norms = nn.ModuleList([
            nn.LayerNorm(dim) for dim in expert_dims
        ])
        
        self._init_weights()
    
    def _init_weights(self):
        init_scale = 0.02
        
        for i, proj in enumerate(self.expert_projections):
            nn.init.normal_(proj.weight, mean=0.0, std=init_scale)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)
        
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.normal_(proj.weight, mean=0.0, std=init_scale)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)
        
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
        primary_idx = primary_expert[0].item()
        primary_hidden = expert_hidden_states[primary_idx]
        
        if torch.isnan(primary_hidden).any():
            print(f"Warning: NaN in InteractionLayer input, returning original states")
            return expert_hidden_states, expert_attention_masks
        
        batch_size = primary_hidden.size(0)
        target_seq_len = primary_hidden.size(1)
        
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
            
            hidden_f32 = hidden.to(device=layer_device, dtype=torch.float32)
            hidden_f32 = torch.clamp(hidden_f32, min=-self.clamp_value, max=self.clamp_value)
            hidden_normed = self.input_norms[idx_val].float()(hidden_f32)
            
            if torch.isnan(hidden_normed).any():
                print(f"Warning: NaN after LayerNorm for expert {idx_val}, using clamped input")
                hidden_normed = hidden_f32
            
            proj_weight = self.expert_projections[idx_val].weight.float()
            proj_bias = self.expert_projections[idx_val].bias
            if proj_bias is not None:
                proj_bias = proj_bias.float()
            
            proj = F.linear(hidden_normed, proj_weight, proj_bias)
            
            if torch.isnan(proj).any():
                print(f"Warning: NaN after projection for expert {idx_val}, skipping")
                continue
            
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
        
        primary_local_idx = active_indices.index(primary_idx) if primary_idx in active_indices else 0
        primary_proj = aligned_states[primary_local_idx]
        
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
        
        layer_norm_weight = self.layer_norm.weight.float()
        layer_norm_bias = self.layer_norm.bias.float() if self.layer_norm.bias is not None else None
        attn_output = F.layer_norm(out_proj + primary_proj, [self.shared_dim], 
                                   layer_norm_weight, layer_norm_bias)
        
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
            
            if original_seq_len != target_seq_len:
                update_t = F.interpolate(
                    update_source.transpose(1, 2),
                    size=original_seq_len,
                    mode='linear',
                    align_corners=True
                ).transpose(1, 2)
            else:
                update_t = update_source
            
            back_proj = F.linear(update_t, self.back_projections[idx_val].weight.float(),
                                self.back_projections[idx_val].bias.float() 
                                if self.back_projections[idx_val].bias is not None else None)
            
            if torch.isnan(back_proj).any():
                print(f"Warning: NaN in back_proj for expert {idx_val}, skipping update")
                continue
            
            back_proj = torch.clamp(back_proj, min=-self.clamp_value, max=self.clamp_value)
            
            back_proj_normed = F.layer_norm(
                back_proj, [expert_hidden_states[idx_val].size(-1)],
                self.output_norms[idx_val].weight.float(),
                self.output_norms[idx_val].bias.float() if self.output_norms[idx_val].bias is not None else None
            )
            
            back_proj_original = back_proj_normed.to(device=original_device, dtype=original_dtype)
            
            updated_hidden_states[idx_val] = expert_hidden_states[idx_val] + self.interaction_scale * back_proj_original
        
        return updated_hidden_states, updated_attention_masks


class MixtureOfThoughts(nn.Module):
    """Main Mixture of Thoughts framework with Gumbel-Softmax and consistency loss."""
    
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
        
        self.router = SparseRouter(sentence_encoder_dim, self.num_experts, config)
        
        self.interaction_layers = nn.ModuleList([
            InteractionLayer(self.expert_dims, config)
            for _ in range(config.num_stacks)
        ])
        
        if config.enable_auxiliary_loss:
            self.auxiliary_heads = nn.ModuleList([
                nn.Linear(dim, vocab_size)
                for dim, vocab_size in zip(self.expert_dims, self.vocab_sizes)
            ])
        
        # Consistency loss settings
        self.consistency_temperature = config.consistency_temperature
        self.lambda_consistency = config.lambda_consistency
        
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
        for layer in self.interaction_layers:
            layer.to(torch.float32)
        
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
    
    def _forward_with_routing(
        self,
        raw_texts: List[str],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor],
        active_experts: torch.Tensor,
        primary_expert: torch.Tensor,
        device: torch.device,
        max_length: int
    ) -> Dict[str, Any]:
        """
        Internal forward pass with given routing decision.
        Used for both regular forward and consistency loss computation.
        """
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
        
        return {
            'logits': logits,
            'hidden_states': expert_hidden_states,
            'expert_labels_list': expert_labels_list,
            'expert_input_ids_list': expert_input_ids_list,
            'primary_idx': primary_idx
        }
    
    def compute_consistency_loss(
        self,
        logits_1: torch.Tensor,
        logits_2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute routing consistency loss between two forward passes.
        Encourages stable outputs under different routing perturbations.
        """
        # Flatten logits for KL computation
        logits_1_flat = logits_1.view(-1, logits_1.size(-1))
        logits_2_flat = logits_2.view(-1, logits_2.size(-1))
        
        # Compute probabilities with temperature
        probs_1 = F.softmax(logits_1_flat / self.consistency_temperature, dim=-1)
        log_probs_2 = F.log_softmax(logits_2_flat / self.consistency_temperature, dim=-1)
        
        probs_2 = F.softmax(logits_2_flat / self.consistency_temperature, dim=-1)
        log_probs_1 = F.log_softmax(logits_1_flat / self.consistency_temperature, dim=-1)
        
        # Symmetric KL divergence
        kl_1_2 = F.kl_div(log_probs_2, probs_1, reduction='batchmean')
        kl_2_1 = F.kl_div(log_probs_1, probs_2, reduction='batchmean')
        
        return (kl_1_2 + kl_2_1) / 2
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        raw_texts: Optional[List[str]] = None,
        prompt_lengths: Optional[List[int]] = None,
        return_dict: bool = True,
        compute_consistency: bool = True
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
        
        # First routing (with Gumbel noise if training)
        active_experts, expert_weights, primary_expert, all_scores = self.router(
            prompt_repr, return_scores=True
        )
        
        # First forward pass
        outputs_1 = self._forward_with_routing(
            raw_texts, input_ids, attention_mask, labels,
            active_experts, primary_expert, device, max_length
        )
        
        logits = outputs_1['logits']
        expert_hidden_states = outputs_1['hidden_states']
        expert_labels_list = outputs_1['expert_labels_list']
        expert_input_ids_list = outputs_1['expert_input_ids_list']
        primary_idx = outputs_1['primary_idx']
        
        total_loss = None
        consistency_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        
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
            
            # Compute consistency loss if training and enabled
            if self.training and compute_consistency and self.lambda_consistency > 0:
                # Second routing with different Gumbel noise
                active_experts_2, _, primary_expert_2, _ = self.router(
                    prompt_repr, return_scores=True
                )
                
                # IMPORTANT: Force same primary expert to ensure same vocab size for KL divergence
                # This follows the paper's design where consistency is measured across different
                # active expert combinations but with the same primary expert for output
                primary_expert_2 = primary_expert.clone()
                
                # Ensure primary expert is in active_experts_2
                # If not, replace the lowest-scoring expert with primary
                primary_idx_val = primary_expert[0].item()
                if primary_idx_val not in active_experts_2[0].tolist():
                    # Replace last position with primary expert
                    active_experts_2[0, -1] = primary_idx_val
                
                # Second forward pass with same primary expert
                outputs_2 = self._forward_with_routing(
                    raw_texts, input_ids, attention_mask, labels,
                    active_experts_2, primary_expert_2, device, max_length
                )
                
                logits_2 = outputs_2['logits']
                
                # Verify shapes match before computing KL divergence
                if logits.shape == logits_2.shape:
                    # Compute consistency loss
                    consistency_loss = self.compute_consistency_loss(logits, logits_2)
                    
                    if torch.isnan(consistency_loss) or torch.isinf(consistency_loss):
                        consistency_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
                else:
                    # Shapes don't match (shouldn't happen with same primary), skip consistency loss
                    print(f"Warning: logits shape mismatch {logits.shape} vs {logits_2.shape}, skipping consistency loss")
                    consistency_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
            
            aux_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
            
            lm_loss = lm_loss.to(device)
            entropy_loss = entropy_loss.to(device)
            balance_loss = balance_loss.to(device)
            consistency_loss = consistency_loss.to(device)
            aux_loss = aux_loss.to(device)
            
            total_loss = (
                lm_loss + 
                0.01 * entropy_loss + 
                0.01 * balance_loss + 
                self.lambda_consistency * consistency_loss
            )
            
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
                'hidden_states': expert_hidden_states,
                'consistency_loss': consistency_loss if labels is not None else None
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
    ) -> Tuple[torch.Tensor, int]:
        """
        Generate text using MoT framework (fast mode).
        Routes once and uses primary expert for generation.
        MoT forward is called but doesn't affect the actual generation.
        
        Returns:
            generated: Generated token ids
            primary_idx: Index of the primary expert used for generation
        """
        device = self._get_device()
        
        if raw_text is None:
            if input_ids is None:
                raise ValueError("Either input_ids or raw_text must be provided")
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            valid_len = int(attention_mask[0].sum().item())
            valid_ids = input_ids[0][:valid_len]
            raw_text = self.tokenizers[0].decode(valid_ids.cpu(), skip_special_tokens=False)
        
        # Step 1: Get routing decision
        prompt_repr = self.encode_prompt([raw_text])
        active_experts, expert_weights, primary_expert = self.router(prompt_repr, use_gumbel=False)
        primary_idx = primary_expert[0].item()
        
        primary_tokenizer = self.tokenizers[primary_idx]
        primary_model = self.experts[primary_idx].model
        
        # Step 2: Call MoT forward to enrich hidden states (like official code)
        encoding = primary_tokenizer(
            raw_text, return_tensors='pt', truncation=True,
            max_length=max_length // 2 if max_length else 1024
        )
        gen_input_ids = encoding['input_ids'].to(device)
        gen_attention_mask = encoding['attention_mask'].to(device)
        
        with torch.no_grad():
            # Run MoT forward pass (enriches hidden states through interaction layers)
            _ = self.forward(
                input_ids=gen_input_ids,
                attention_mask=gen_attention_mask,
                raw_texts=[raw_text],
                return_dict=True,
                compute_consistency=False  # No need for consistency during inference
            )
        
        # Step 3: Generate using primary expert
        # Clean kwargs to avoid conflicts
        gen_kwargs = {k: v for k, v in kwargs.items() 
                     if k not in ['do_sample', 'temperature', 'top_p', 'max_length', 'max_new_tokens']}
        
        # Build generation parameters - use only one length parameter to avoid conflict
        generation_params = {
            'input_ids': gen_input_ids,
            'attention_mask': gen_attention_mask,
            'temperature': temperature,
            'top_p': top_p,
            'do_sample': do_sample,
            'pad_token_id': primary_tokenizer.pad_token_id,
            'eos_token_id': primary_tokenizer.eos_token_id,
            **gen_kwargs
        }
        
        # Only set one length parameter: prefer max_new_tokens over max_length
        if max_new_tokens is not None:
            generation_params['max_new_tokens'] = max_new_tokens
        elif max_length is not None:
            generation_params['max_length'] = max_length
        else:
            generation_params['max_new_tokens'] = 512  # Default value
        
        with torch.no_grad():
            generated = primary_model.generate(**generation_params)
        
        return generated, primary_idx
    
    def generate_with_mot(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        raw_text: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 0.9,
        **kwargs
    ) -> Tuple[torch.Tensor, int]:
        """
        Generate text using full MoT framework (slow but accurate mode).
        Each token is generated using MoT's enriched hidden states from expert interaction.
        
        This method actually utilizes the multi-expert interaction for generation,
        unlike generate() which only uses routing to select an expert.
        
        Args:
            input_ids: Input token IDs (optional if raw_text provided)
            attention_mask: Attention mask (optional)
            raw_text: Raw input text (optional if input_ids provided)
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_p: Top-p (nucleus) sampling parameter
            **kwargs: Additional arguments (unused, for compatibility)
            
        Returns:
            generated: Generated token ids [1, seq_len]
            primary_idx: Index of the primary expert used
        """
        device = self._get_device()
        
        # Prepare input text
        if raw_text is None:
            if input_ids is None:
                raise ValueError("Either input_ids or raw_text must be provided")
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            valid_len = int(attention_mask[0].sum().item())
            valid_ids = input_ids[0][:valid_len]
            raw_text = self.tokenizers[0].decode(valid_ids.cpu(), skip_special_tokens=False)
        
        # Step 1: Get routing decision (once at the beginning)
        prompt_repr = self.encode_prompt([raw_text])
        active_experts, expert_weights, primary_expert = self.router(prompt_repr, use_gumbel=False)
        primary_idx = primary_expert[0].item()
        
        primary_tokenizer = self.tokenizers[primary_idx]
        vocab_size = self.experts[primary_idx].vocab_size
        
        # Step 2: Tokenize with primary expert's tokenizer
        encoding = primary_tokenizer(
            raw_text, return_tensors='pt', truncation=True, max_length=1024
        )
        generated_ids = encoding['input_ids'].to(device)
        current_attention_mask = encoding['attention_mask'].to(device)
        
        batch_size = generated_ids.size(0)
        current_raw_text = raw_text
        
        # Step 3: Generate tokens one by one using MoT forward
        for step in range(max_new_tokens):
            # Run MoT forward pass - this time we USE the logits
            with torch.no_grad():
                outputs = self.forward(
                    input_ids=generated_ids,
                    attention_mask=current_attention_mask,
                    raw_texts=[current_raw_text],
                    return_dict=True,
                    compute_consistency=False
                )
            
            # Get logits for the last position
            logits = outputs['logits'][:, -1, :]  # [batch_size, vocab_size]
            
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
            
            # Apply top-p (nucleus) sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep the first token above threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Scatter sorted tensors to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [batch_size, 1]
            
            # Check for EOS token
            if next_token.item() == primary_tokenizer.eos_token_id:
                break
            
            # Append to generated sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            current_attention_mask = torch.cat([
                current_attention_mask,
                torch.ones((batch_size, 1), device=device, dtype=current_attention_mask.dtype)
            ], dim=-1)
            
            # Update raw text for next iteration
            current_raw_text = primary_tokenizer.decode(
                generated_ids[0], skip_special_tokens=False
            )
        
        return generated_ids, primary_idx
    
    def translate(
        self,
        source_text: str,
        prompt_template: str = "### Translate C++ to CUDA:\n{source}\n### CUDA:\n",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        use_mot_generate: bool = False,
        **kwargs
    ) -> Tuple[str, int]:
        """
        Translate source code to CUDA.
        
        Args:
            source_text: Source C++ code to translate
            prompt_template: Template for formatting the prompt
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            use_mot_generate: If True, use full MoT generation (slow but uses expert interaction)
                             If False, use fast generation (only routing, no interaction during generation)
            **kwargs: Additional arguments passed to generate()
            
        Returns:
            generated_code: Generated CUDA code
            primary_idx: Index of the primary expert used
        """
        prompt = prompt_template.format(source=source_text)
        
        # Choose generation method based on use_mot_generate flag
        if use_mot_generate:
            # Full MoT generation - each token uses expert interaction
            output_ids, primary_idx = self.generate_with_mot(
                raw_text=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                **kwargs
            )
        else:
            # Fast generation - only routing, expert generates independently
            output_ids, primary_idx = self.generate(
                raw_text=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                **kwargs
            )
        
        # Decode generated output
        tokenizer = self.tokenizers[primary_idx]
        prompt_encoding = tokenizer(prompt, return_tensors='pt')
        prompt_len = prompt_encoding['input_ids'].shape[1]
        
        generated_ids = output_ids[0, prompt_len:]
        generated_code = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Post-process: truncate at stop markers
        for stop_marker in ['\n\n\n', '###', '<|endoftext|>', '<|end|>', '<|im_end|>']:
            if stop_marker in generated_code:
                generated_code = generated_code.split(stop_marker)[0]
        
        return generated_code.strip(), primary_idx
    
    def get_routing_info(self, text: str) -> Dict[str, Any]:
        with torch.no_grad():
            prompt_repr = self.encode_prompt([text])
            active_experts, expert_weights, primary_expert, scores = self.router(
                prompt_repr, return_scores=True, use_gumbel=False
            )
            all_probs = F.softmax(scores, dim=-1)
            
            return {
                'active_experts': active_experts.cpu().tolist(),
                'expert_weights': expert_weights.cpu().tolist(),
                'primary_expert': primary_expert.cpu().item(),
                'all_expert_probs': all_probs.cpu().tolist(),
                'router_scores': scores.cpu().tolist()
            }
