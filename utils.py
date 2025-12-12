"""
Utility functions for Mixture of Thoughts framework.
Updated with 8-bit and 4-bit quantization support.

MODIFICATION: Added single_gpu option to avoid device_map='auto' which causes
issues with manual layer-by-layer forward passes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Tuple, Any, Union
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import numpy as np
from dataclasses import dataclass
import json
import os


@dataclass
class ExpertConfig:
    """Configuration for individual expert models."""
    model_name: str
    model_type: str  # 'causal_lm', 'masked_lm', 'seq2seq'
    tokenizer_name: Optional[str] = None
    device_map: Optional[str] = None
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    torch_dtype: Optional[torch.dtype] = None
    single_gpu: bool = False  # If True, load entire model on one GPU (no device_map='auto')
    target_device: Optional[str] = None  # e.g., 'cuda:0' for single_gpu mode


class ExpertLoader:
    """Utility class for loading and managing expert models."""
    
    @staticmethod
    def load_expert(config: ExpertConfig, cache_dir: Optional[str] = None) -> Tuple[Any, Any]:
        """
        Load an expert model and its tokenizer.
        
        Args:
            config: Configuration for the expert
            cache_dir: Optional cache directory for models
            
        Returns:
            model: Loaded model
            tokenizer: Loaded tokenizer
        """
        tokenizer_name = config.tokenizer_name or config.model_name
        
        # Set cache directory
        cache_dir = cache_dir or os.path.expanduser("~/.cache/huggingface/hub")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            cache_dir=cache_dir,
            local_files_only=False,
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Build model kwargs
        model_kwargs = {
            'cache_dir': cache_dir,
            'local_files_only': False,
            'trust_remote_code': True
        }
        
        # Handle quantization and device placement
        if config.load_in_4bit:
            # 4-bit quantization config
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            model_kwargs['quantization_config'] = quantization_config
            
            # CRITICAL: For single_gpu mode, use specific device instead of 'auto'
            if config.single_gpu:
                target_device = config.target_device or 'cuda:0'
                model_kwargs['device_map'] = {'': target_device}
                print(f"  Using 4-bit quantization on {target_device} (single_gpu mode)")
            else:
                model_kwargs['device_map'] = 'auto'
                print(f"  Using 4-bit quantization with device_map='auto'")
                
        elif config.load_in_8bit:
            # 8-bit quantization
            model_kwargs['load_in_8bit'] = True
            
            if config.single_gpu:
                target_device = config.target_device or 'cuda:0'
                model_kwargs['device_map'] = {'': target_device}
                print(f"  Using 8-bit quantization on {target_device} (single_gpu mode)")
            else:
                model_kwargs['device_map'] = 'auto'
                print(f"  Using 8-bit quantization with device_map='auto'")
        else:
            # Standard loading (no quantization)
            if config.single_gpu:
                # Don't use device_map, we'll move the model manually
                if config.torch_dtype:
                    model_kwargs['torch_dtype'] = config.torch_dtype
                print(f"  Loading full precision model (single_gpu mode)")
            else:
                if config.device_map:
                    model_kwargs['device_map'] = config.device_map
                if config.torch_dtype:
                    model_kwargs['torch_dtype'] = config.torch_dtype
        
        # Load model
        try:
            if config.model_type == 'causal_lm':
                # Try local cache first
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        config.model_name,
                        **{k: v for k, v in model_kwargs.items() if k != 'local_files_only'},
                        local_files_only=True
                    )
                    print(f"✓ Loaded {config.model_name} from cache")
                except:
                    print(f"↓ Downloading {config.model_name} (not in cache)")
                    model = AutoModelForCausalLM.from_pretrained(
                        config.model_name,
                        **model_kwargs
                    )
                    print(f"✓ Downloaded {config.model_name}")
            elif config.model_type == 'masked_lm':
                try:
                    model = AutoModel.from_pretrained(
                        config.model_name,
                        **{k: v for k, v in model_kwargs.items() if k != 'local_files_only'},
                        local_files_only=True
                    )
                    print(f"✓ Loaded {config.model_name} from cache")
                except:
                    print(f"↓ Downloading {config.model_name} (not in cache)")
                    model = AutoModel.from_pretrained(
                        config.model_name,
                        **model_kwargs
                    )
            else:
                raise ValueError(f"Unsupported model type: {config.model_type}")
                
            # For single_gpu mode without quantization, move to target device
            if config.single_gpu and not config.load_in_4bit and not config.load_in_8bit:
                target_device = config.target_device or 'cuda:0'
                model = model.to(target_device)
                print(f"  Moved model to {target_device}")
                
        except Exception as e:
            print(f"Error loading model {config.model_name}: {e}")
            raise
        
        return model, tokenizer
    
    @staticmethod
    def load_multiple_experts(
        configs: List[ExpertConfig],
        cache_dir: Optional[str] = None
    ) -> Tuple[List[Any], List[Any]]:
        """
        Load multiple expert models.
        
        Args:
            configs: List of expert configurations
            cache_dir: Optional cache directory for models
            
        Returns:
            models: List of loaded models
            tokenizers: List of loaded tokenizers
        """
        models = []
        tokenizers = []
        
        cache_dir = cache_dir or os.path.expanduser("~/.cache/huggingface/hub")
        print(f"Loading models from cache: {cache_dir}")
        
        for i, config in enumerate(configs, 1):
            print(f"\nLoading expert {i}/{len(configs)}: {config.model_name}")
            model, tokenizer = ExpertLoader.load_expert(config, cache_dir=cache_dir)
            models.append(model)
            tokenizers.append(tokenizer)
        
        print(f"\n✓ Successfully loaded {len(models)} expert models")
        return models, tokenizers


class TokenAligner:
    """Utility for aligning tokens across different tokenizers."""
    
    def __init__(self, tokenizers: List[AutoTokenizer]):
        self.tokenizers = tokenizers
        self.num_experts = len(tokenizers)
    
    def align_tokens(
        self,
        text: str,
        max_length: int = 512
    ) -> Dict[str, torch.Tensor]:
        """Tokenize text with all tokenizers and align the results."""
        all_input_ids = []
        all_attention_masks = []
        
        for tokenizer in self.tokenizers:
            encoding = tokenizer(
                text,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            all_input_ids.append(encoding['input_ids'])
            all_attention_masks.append(encoding['attention_mask'])
        
        input_ids = torch.cat(all_input_ids, dim=0)
        attention_masks = torch.cat(all_attention_masks, dim=0)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_masks
        }
    
    def decode_from_expert(
        self,
        token_ids: torch.Tensor,
        expert_idx: int,
        skip_special_tokens: bool = True
    ) -> str:
        """Decode token IDs using specific expert's tokenizer."""
        tokenizer = self.tokenizers[expert_idx]
        text = tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens
        )
        return text


class ExpertProfiler:
    """Utility for profiling and analyzing expert behaviors."""
    
    def __init__(self, num_experts: int):
        self.num_experts = num_experts
        self.reset_stats()
    
    def reset_stats(self):
        """Reset all statistics."""
        self.selection_counts = np.zeros(self.num_experts)
        self.primary_counts = np.zeros(self.num_experts)
        self.avg_scores = []
        self.routing_history = []
    
    def update(
        self,
        active_experts: torch.Tensor,
        primary_expert: torch.Tensor,
        scores: Optional[torch.Tensor] = None
    ):
        """Update statistics with new routing decision."""
        for idx in active_experts.cpu().numpy().flatten():
            self.selection_counts[idx] += 1
        
        primary_idx = primary_expert.cpu().item()
        self.primary_counts[primary_idx] += 1
        
        self.routing_history.append({
            'active': active_experts.cpu().tolist(),
            'primary': primary_idx
        })
        
        if scores is not None:
            self.avg_scores.append(scores.cpu().numpy())
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about expert usage."""
        total_selections = self.selection_counts.sum()
        total_primary = self.primary_counts.sum()
        
        stats = {
            'selection_frequency': self.selection_counts / (total_selections + 1e-8),
            'primary_frequency': self.primary_counts / (total_primary + 1e-8),
            'selection_counts': self.selection_counts.tolist(),
            'primary_counts': self.primary_counts.tolist(),
            'total_routing_decisions': len(self.routing_history)
        }
        
        if self.avg_scores:
            avg_scores_array = np.mean(self.avg_scores, axis=0)
            stats['average_scores'] = avg_scores_array.tolist()
        
        selection_entropy = self._compute_entropy(stats['selection_frequency'])
        primary_entropy = self._compute_entropy(stats['primary_frequency'])
        
        stats['selection_entropy'] = selection_entropy
        stats['primary_entropy'] = primary_entropy
        
        return stats
    
    def _compute_entropy(self, probs: np.ndarray) -> float:
        """Compute entropy of probability distribution."""
        probs = probs + 1e-10
        entropy = -np.sum(probs * np.log(probs))
        return float(entropy)
    
    def save_profile(self, filepath: str):
        """Save profiling results to file."""
        stats = self.get_statistics()
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Expert profile saved to {filepath}")


class GradientTracker:
    """Utility for tracking gradient statistics during training."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.reset_stats()
    
    def reset_stats(self):
        """Reset gradient statistics."""
        self.grad_norms = {}
        self.grad_means = {}
        self.grad_stds = {}
    
    def update(self):
        """Update gradient statistics after backward pass."""
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad = param.grad.data
                
                grad_norm = grad.norm().item()
                grad_mean = grad.mean().item()
                grad_std = grad.std().item()
                
                if name not in self.grad_norms:
                    self.grad_norms[name] = []
                    self.grad_means[name] = []
                    self.grad_stds[name] = []
                
                self.grad_norms[name].append(grad_norm)
                self.grad_means[name].append(grad_mean)
                self.grad_stds[name].append(grad_std)
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary of gradient statistics."""
        summary = {}
        
        for name in self.grad_norms.keys():
            summary[name] = {
                'avg_norm': np.mean(self.grad_norms[name]),
                'max_norm': np.max(self.grad_norms[name]),
                'avg_mean': np.mean(self.grad_means[name]),
                'avg_std': np.mean(self.grad_stds[name])
            }
        
        return summary
    
    def check_gradient_health(self) -> Dict[str, bool]:
        """Check for gradient issues."""
        issues = {
            'vanishing_gradients': False,
            'exploding_gradients': False,
            'dead_neurons': False
        }
        
        for name, norms in self.grad_norms.items():
            avg_norm = np.mean(norms)
            
            if avg_norm < 1e-7:
                issues['vanishing_gradients'] = True
            
            if avg_norm > 100:
                issues['exploding_gradients'] = True
            
            if all(n < 1e-10 for n in norms[-10:]):
                issues['dead_neurons'] = True
        
        return issues


class CheckpointManager:
    """Utility for managing model checkpoints."""
    
    def __init__(self, checkpoint_dir: str, max_checkpoints: int = 5):
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.checkpoint_history = []
    
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False
    ):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'timestamp': str(np.datetime64('now'))
        }
        
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f'checkpoint_epoch_{epoch}.pt'
        )
        torch.save(checkpoint, checkpoint_path)
        self.checkpoint_history.append(checkpoint_path)
        
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_checkpoint.pt')
            torch.save(checkpoint, best_path)
            print(f"Saved best checkpoint at epoch {epoch}")
        
        if len(self.checkpoint_history) > self.max_checkpoints:
            old_checkpoint = self.checkpoint_history.pop(0)
            if os.path.exists(old_checkpoint):
                os.remove(old_checkpoint)
                print(f"Removed old checkpoint: {old_checkpoint}")
    
    def load_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        checkpoint_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Load model checkpoint."""
        if checkpoint_path is None:
            checkpoint_path = os.path.join(self.checkpoint_dir, 'best_checkpoint.pt')
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        
        return checkpoint
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get path to latest checkpoint."""
        if self.checkpoint_history:
            return self.checkpoint_history[-1]
        
        checkpoints = [
            f for f in os.listdir(self.checkpoint_dir)
            if f.startswith('checkpoint_epoch_') and f.endswith('.pt')
        ]
        
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
            return os.path.join(self.checkpoint_dir, checkpoints[-1])
        
        return None


def compute_model_size(model: nn.Module) -> Dict[str, Any]:
    """
    Compute model size and parameter statistics.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with size statistics
    """
    total_params = 0
    trainable_params = 0
    param_groups = {}
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        
        if param.requires_grad:
            trainable_params += num_params
        
        component = name.split('.')[0]
        if component not in param_groups:
            param_groups[component] = {'total': 0, 'trainable': 0}
        
        param_groups[component]['total'] += num_params
        if param.requires_grad:
            param_groups[component]['trainable'] += num_params
    
    stats = {
        'total_params_M': total_params / 1e6,
        'trainable_params_M': trainable_params / 1e6,
        'frozen_params_M': (total_params - trainable_params) / 1e6,
        'trainable_percentage': (trainable_params / total_params) * 100 if total_params > 0 else 0,
        'param_groups': {
            k: {
                'total_M': v['total'] / 1e6,
                'trainable_M': v['trainable'] / 1e6
            }
            for k, v in param_groups.items()
        }
    }
    
    return stats


def get_gpu_memory_info() -> Dict[str, Any]:
    """Get GPU memory information."""
    if not torch.cuda.is_available():
        return {'available': False}
    
    info = {
        'available': True,
        'device_count': torch.cuda.device_count(),
        'devices': []
    }
    
    for i in range(torch.cuda.device_count()):
        device_info = {
            'index': i,
            'name': torch.cuda.get_device_name(i),
            'total_memory_GB': torch.cuda.get_device_properties(i).total_memory / 1e9,
            'allocated_memory_GB': torch.cuda.memory_allocated(i) / 1e9,
            'reserved_memory_GB': torch.cuda.memory_reserved(i) / 1e9,
            'free_memory_GB': (torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)) / 1e9
        }
        info['devices'].append(device_info)
    
    return info


def print_gpu_memory():
    """Print GPU memory usage."""
    info = get_gpu_memory_info()
    
    if not info['available']:
        print("CUDA not available")
        return
    
    print(f"\nGPU Memory Usage ({info['device_count']} device(s)):")
    print("-" * 50)
    
    for device in info['devices']:
        print(f"  GPU {device['index']}: {device['name']}")
        print(f"    Total: {device['total_memory_GB']:.2f} GB")
        print(f"    Allocated: {device['allocated_memory_GB']:.2f} GB")
        print(f"    Free: {device['free_memory_GB']:.2f} GB")
    print("-" * 50)
