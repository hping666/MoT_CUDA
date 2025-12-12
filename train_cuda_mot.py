"""
Training script for C++ to CUDA code translation using Mixture of Thoughts (MoT).

Uses three expert models:
1. Qwen2.5-Coder-7B
2. HPC-Coder-v2-6.7B  
3. StarCoder2-7B

Only trains the router and interaction layers (experts are frozen).

FIXES:
- Updated to pass raw_texts to model for multi-tokenizer support
- Fixed batch handling for different vocab sizes across experts
"""

import os
import sys
import json
import argparse
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

# Local imports
from mixture_of_thoughts import MixtureOfThoughts, MoTConfig
from cuda_dataset import (
    CppCudaDataset, 
    load_cpp_cuda_data, 
    DataCollatorForCppCuda,
    create_dataloaders,
    get_dataset_statistics
)
from cuda_evaluation import CudaCodeEvaluator, evaluate_generation
from utils import ExpertConfig, ExpertLoader, compute_model_size


# =============================================================================
# DEFAULT CONFIGURATION - Modify these values directly as needed
# =============================================================================
DEFAULT_CONFIG = {
    # Data settings
    'data_path': './cpp_cuda_train_synthetic.jsonl',
    'train_ratio': 0.8,
    'test_ratio': 0.1,
    
    # Model settings
    'use_8bit': False,
    'use_4bit': True,
    'num_stacks': 4,
    'top_k': 2,
    'single_gpu': True,  # Load all models on single GPU (recommended for MoT)
    
    # Training settings
    'batch_size': 1,
    'gradient_accumulation_steps': 8,
    'num_epochs': 3,
    'learning_rate': 5e-6,  # Reduced from 1e-4 for stability
    'warmup_steps': 200,
    'max_length': 1024,  # Reduced from 2048 for memory
    'max_new_tokens': 512,
    'gradient_clip': 0.3,  # Reduced from 1.0 for stability
    
    # Evaluation settings
    'eval_steps': 100,
    'eval_samples': 50,
    'eval_only': False,
    
    # Output settings
    'output_dir': './cuda_mot_output',
    'checkpoint_path': None,
    'save_steps': 500,
    
    # Other settings
    'seed': 42,
    'num_workers': 0,
    'logging_steps': 10,
}

# =============================================================================
# EXPERT MODEL CONFIGURATION - Modify model list here
# =============================================================================
# EXPERT_MODELS = [
#     {
#         'name': 'Qwen/Qwen2.5-Coder-7B',
#         'description': 'Qwen2.5 Coder 7B - Strong general code generation',
#     },
#     {
#         'name': 'hpcgroup/hpc-coder-v2-6.7b',
#         'description': 'HPC-Coder-v2 - Specialized for HPC/parallel code',
#     },
#     {
#         'name': 'bigcode/starcoder2-7b',
#         'description': 'StarCoder2 7B - Multi-language code generation',
#     },
# ]

EXPERT_MODELS = [
    {
        'name': 'Qwen/Qwen2.5-Coder-1.5B',  
        'description': 'Qwen2.5 Coder 1.5B - Lightweight code generation',
    },
    {
        'name': 'hpcgroup/hpc-coder-v2-1.3b',
        'description': 'HPC-Coder-v2 1.3B - Lightweight HPC code',
    },
    {
        'name': 'bigcode/starcoder2-3b',
        'description': 'StarCoder2 3B - Lightweight multi-language code',
    },
]


def parse_args():
    """Parse command line arguments with defaults from DEFAULT_CONFIG."""
    parser = argparse.ArgumentParser(
        description='Train MoT for C++ to CUDA code translation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument(
        '--data_path',
        type=str,
        default=DEFAULT_CONFIG['data_path'],
        help='Path to the JSONL dataset file'
    )
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=DEFAULT_CONFIG['train_ratio'],
        help='Ratio of data to use for training (0.0 to 1.0+)'
    )
    parser.add_argument(
        '--test_ratio',
        type=float,
        default=DEFAULT_CONFIG['test_ratio'],
        help='Ratio of data to use for testing (0.0 to 1.0+)'
    )
    
    # Model arguments
    parser.add_argument(
        '--use_8bit',
        action='store_true',
        default=DEFAULT_CONFIG['use_8bit'],
        help='Load models in 8-bit quantization to save memory'
    )
    parser.add_argument(
        '--use_4bit',
        action='store_true',
        default=DEFAULT_CONFIG['use_4bit'],
        help='Load models in 4-bit quantization to save more memory'
    )
    parser.add_argument(
        '--num_stacks',
        type=int,
        default=DEFAULT_CONFIG['num_stacks'],
        help='Number of stacks to divide expert layers into'
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=DEFAULT_CONFIG['top_k'],
        help='Number of experts to activate per forward pass'
    )
    parser.add_argument(
        '--single_gpu',
        action='store_true',
        default=DEFAULT_CONFIG['single_gpu'],
        help='Load all models on single GPU (recommended). Disable for multi-GPU DDP.'
    )
    parser.add_argument(
        '--no_single_gpu',
        action='store_true',
        help='Disable single_gpu mode (use device_map="auto" for multi-GPU)'
    )
    
    # Training arguments
    parser.add_argument(
        '--batch_size',
        type=int,
        default=DEFAULT_CONFIG['batch_size'],
        help='Training batch size'
    )
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=DEFAULT_CONFIG['gradient_accumulation_steps'],
        help='Number of gradient accumulation steps'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=DEFAULT_CONFIG['num_epochs'],
        help='Number of training epochs'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=DEFAULT_CONFIG['learning_rate'],
        help='Learning rate'
    )
    parser.add_argument(
        '--warmup_steps',
        type=int,
        default=DEFAULT_CONFIG['warmup_steps'],
        help='Number of warmup steps'
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=DEFAULT_CONFIG['max_length'],
        help='Maximum sequence length for training'
    )
    parser.add_argument(
        '--max_new_tokens',
        type=int,
        default=DEFAULT_CONFIG['max_new_tokens'],
        help='Maximum new tokens to generate during evaluation'
    )
    parser.add_argument(
        '--gradient_clip',
        type=float,
        default=DEFAULT_CONFIG['gradient_clip'],
        help='Gradient clipping value'
    )
    
    # Evaluation arguments
    parser.add_argument(
        '--eval_steps',
        type=int,
        default=DEFAULT_CONFIG['eval_steps'],
        help='Evaluate every N steps'
    )
    parser.add_argument(
        '--eval_samples',
        type=int,
        default=DEFAULT_CONFIG['eval_samples'],
        help='Number of samples to evaluate during training'
    )
    parser.add_argument(
        '--eval_only',
        action='store_true',
        default=DEFAULT_CONFIG['eval_only'],
        help='Only run evaluation (no training)'
    )
    
    # Output arguments
    parser.add_argument(
        '--output_dir',
        type=str,
        default=DEFAULT_CONFIG['output_dir'],
        help='Directory to save outputs'
    )
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default=DEFAULT_CONFIG['checkpoint_path'],
        help='Path to checkpoint to load'
    )
    parser.add_argument(
        '--save_steps',
        type=int,
        default=DEFAULT_CONFIG['save_steps'],
        help='Save checkpoint every N steps'
    )
    
    # Other arguments
    parser.add_argument(
        '--seed',
        type=int,
        default=DEFAULT_CONFIG['seed'],
        help='Random seed'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=DEFAULT_CONFIG['num_workers'],
        help='Number of data loading workers'
    )
    parser.add_argument(
        '--logging_steps',
        type=int,
        default=DEFAULT_CONFIG['logging_steps'],
        help='Log every N steps'
    )
    
    return parser.parse_args()


def load_expert_models(args) -> Tuple[List[Any], List[Any]]:
    """Load the three expert models."""
    print("\n" + "=" * 60)
    print("Loading Expert Models")
    print("=" * 60)
    
    # Determine single_gpu mode
    single_gpu = args.single_gpu and not args.no_single_gpu
    
    if single_gpu:
        print("  Mode: Single GPU (all models on cuda:0)")
        print("  This is recommended for MoT stack-based forward.")
    else:
        print("  Mode: Multi-GPU (device_map='auto')")
        print("  WARNING: This may cause device mismatch errors with stack-based forward!")
    
    expert_configs = []
    
    for model_info in EXPERT_MODELS:
        config = ExpertConfig(
            model_name=model_info['name'],
            model_type='causal_lm',
            torch_dtype=torch.float16,
            load_in_8bit=args.use_8bit,
            load_in_4bit=args.use_4bit,
            single_gpu=single_gpu,
            target_device='cuda:0',
        )
        expert_configs.append(config)
        print(f"  - {model_info['name']}: {model_info['description']}")
    
    print()
    
    # Load models
    models, tokenizers = ExpertLoader.load_multiple_experts(expert_configs)
    
    return models, tokenizers


def setup_mot_model(
    expert_models: List[Any],
    tokenizers: List[Any],
    args
) -> MixtureOfThoughts:
    """Initialize the MoT model."""
    print("\n" + "=" * 60)
    print("Initializing MoT Framework")
    print("=" * 60)
    
    # Get hidden dimensions from models
    hidden_dims = []
    for model in expert_models:
        if hasattr(model.config, 'hidden_size'):
            hidden_dims.append(model.config.hidden_size)
        else:
            hidden_dims.append(4096)  # Default for 7B models
    
    # Use the minimum hidden dim for shared space
    shared_dim = min(hidden_dims)
    
    config = MoTConfig(
        num_stacks=args.num_stacks,
        top_k=args.top_k,
        shared_dim=shared_dim,
        router_hidden_dim=512,
        router_temperature=1.0,
        interaction_heads=8,
        enable_auxiliary_loss=True,
        dropout_rate=0.1,
    )
    
    print(f"  MoT Configuration:")
    print(f"    - Number of stacks: {config.num_stacks}")
    print(f"    - Top-K experts: {config.top_k}")
    print(f"    - Shared dimension: {config.shared_dim}")
    print(f"    - Interaction heads: {config.interaction_heads}")
    
    # Initialize model
    model = MixtureOfThoughts(
        expert_models=expert_models,
        tokenizers=tokenizers,
        config=config
    )
    
    # Print model statistics
    stats = compute_model_size(model)
    print(f"\n  Model Statistics:")
    print(f"    - Total parameters: {stats['total_params_M']:.2f}M")
    print(f"    - Trainable parameters: {stats['trainable_params_M']:.2f}M")
    print(f"    - Trainable percentage: {stats['trainable_percentage']:.2f}%")
    
    return model


def setup_optimizer(model: nn.Module, args) -> Tuple[torch.optim.Optimizer, Any]:
    """Setup optimizer with different learning rates for different components."""
    # Collect trainable parameters by component
    router_params = []
    interaction_params = []
    auxiliary_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'router' in name:
                router_params.append(param)
            elif 'interaction' in name:
                interaction_params.append(param)
            elif 'auxiliary' in name:
                auxiliary_params.append(param)
            else:
                other_params.append(param)
    
    # Different learning rates for different components
    param_groups = []
    if router_params:
        param_groups.append({'params': router_params, 'lr': args.learning_rate * 2.0, 'name': 'router'})
    if interaction_params:
        param_groups.append({'params': interaction_params, 'lr': args.learning_rate, 'name': 'interaction'})
    if auxiliary_params:
        param_groups.append({'params': auxiliary_params, 'lr': args.learning_rate * 0.5, 'name': 'auxiliary'})
    if other_params:
        param_groups.append({'params': other_params, 'lr': args.learning_rate, 'name': 'other'})
    
    optimizer = AdamW(param_groups, weight_decay=0.01)
    
    # Setup scheduler
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=args.warmup_steps,
        T_mult=2
    )
    
    return optimizer, scheduler


def train_step(
    model: nn.Module,
    batch: Dict[str, Any],
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    args,
    step: int,
    accumulation_counter: int
) -> Dict[str, float]:
    """Perform a single training step with multi-tokenizer support."""
    model.train()
    device = next(model.parameters()).device
    
    # Get raw texts from batch (required for multi-tokenizer support)
    raw_texts = batch.get('raw_full_text', None)
    prompt_lengths = batch.get('prompt_length', None)
    
    if raw_texts is None:
        raise ValueError("Batch must contain 'raw_full_text' for multi-tokenizer support")
    
    # Move tensors to device (these are from primary tokenizer, used for reference)
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    
    # Forward pass with raw texts
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        raw_texts=raw_texts,
        prompt_lengths=prompt_lengths,
        return_dict=True
    )
    
    loss = outputs['loss']
    
    # Check for NaN loss
    if loss is None or torch.isnan(loss) or torch.isinf(loss):
        print(f"Warning: Skipping step {step} due to NaN/Inf loss")
        optimizer.zero_grad()
        return {
            'loss': float('nan'),
            'skipped': True
        }
    
    # Scale loss for gradient accumulation
    scaled_loss = loss / args.gradient_accumulation_steps
    
    # Backward pass
    scaled_loss.backward()
    
    metrics = {
        'loss': loss.item(),
    }
    
    # Optimizer step (only after accumulation)
    if (accumulation_counter + 1) % args.gradient_accumulation_steps == 0:
        # Check for NaN gradients before stepping
        has_nan_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                has_nan_grad = True
                break
        
        if has_nan_grad:
            print(f"Warning: NaN gradients detected at step {step}, skipping optimizer step")
            optimizer.zero_grad()
            metrics['skipped'] = True
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            metrics['lr'] = optimizer.param_groups[0]['lr']
    
    # Add routing information
    if 'router_scores' in outputs:
        router_scores = outputs['router_scores']
        probs = torch.softmax(router_scores, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean().item()
        metrics['router_entropy'] = entropy
    
    if 'primary_expert' in outputs:
        metrics['primary_expert'] = outputs['primary_expert'][0].item()
    
    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    test_dataloader: DataLoader,
    tokenizers: List[Any],
    args,
    num_samples: Optional[int] = None
) -> Dict[str, float]:
    """Evaluate the model on test set."""
    model.eval()
    
    predictions = []
    references = []
    expert_usage = {}
    
    evaluator = CudaCodeEvaluator()
    
    # Limit samples if specified
    max_samples = num_samples or len(test_dataloader)
    
    print(f"\nEvaluating on {min(max_samples, len(test_dataloader))} samples...")
    
    for i, batch in enumerate(tqdm(test_dataloader, desc="Evaluating", total=min(max_samples, len(test_dataloader)))):
        if i >= max_samples:
            break
        
        # Get raw text for generation
        raw_prompt = batch.get('raw_prompt', None)
        if raw_prompt is None:
            # Fallback: use cpp_code to create prompt
            cpp_code = batch['cpp_code'][0]
            raw_prompt = [f"### Translate C++ to CUDA:\n{cpp_code}\n### CUDA:\n"]
        
        reference_cuda = batch['cuda_code'][0]
        
        # Generate CUDA code using the translate method
        try:
            generated_cuda, primary_idx = model.translate(
                source_text=batch['cpp_code'][0],
                max_new_tokens=args.max_new_tokens,
                temperature=0.7,
                top_p=0.95,
            )
            
            # Track expert usage
            expert_usage[primary_idx] = expert_usage.get(primary_idx, 0) + 1
            
            predictions.append(generated_cuda)
            references.append(reference_cuda)
            
        except Exception as e:
            print(f"  Error generating for sample {i}: {e}")
            predictions.append("")
            references.append(reference_cuda)
    
    # Compute metrics
    if predictions:
        results = evaluator.evaluate_all(predictions, references)
        results['expert_usage'] = expert_usage
    else:
        results = {'error': 'No predictions generated'}
    
    return results


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    step: int,
    epoch: int,
    best_metric: float,
    args,
    is_best: bool = False
):
    """Save model checkpoint."""
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save only trainable components (router, interaction, auxiliary)
    save_state_dict = {}
    for name, param in model.state_dict().items():
        if 'router' in name or 'interaction' in name or 'auxiliary' in name:
            save_state_dict[name] = param
        # Also save sentence_encoder if needed
        if 'sentence_encoder' in name:
            save_state_dict[name] = param
    
    checkpoint = {
        'step': step,
        'epoch': epoch,
        'model_state_dict': save_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_metric': best_metric,
        'args': vars(args),
    }
    
    # Save regular checkpoint
    checkpoint_path = os.path.join(args.output_dir, f'checkpoint-{step}.pt')
    torch.save(checkpoint, checkpoint_path)
    print(f"  Saved checkpoint: {checkpoint_path}")
    
    # Save best model
    if is_best:
        best_path = os.path.join(args.output_dir, 'best_model.pt')
        torch.save(checkpoint, best_path)
        print(f"  Saved best model: {best_path}")


def load_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device) -> Dict:
    """Load model checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load only the saved parameters (router, interaction, auxiliary)
    model_state = model.state_dict()
    for name, param in checkpoint['model_state_dict'].items():
        if name in model_state:
            model_state[name] = param
    
    model.load_state_dict(model_state)
    return checkpoint


def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    tokenizers: List[Any],
    args
):
    """Main training loop."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"\nUsing device: {device}")
    
    # Setup optimizer and scheduler
    optimizer, scheduler = setup_optimizer(model, args)
    
    # Training state
    global_step = 0
    best_bleu = 0.0
    accumulation_counter = 0
    
    # Load checkpoint if specified
    if args.checkpoint_path:
        checkpoint = load_checkpoint(model, args.checkpoint_path, device)
        global_step = checkpoint.get('step', 0)
        best_bleu = checkpoint.get('best_metric', 0.0)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)
    print(f"  Total epochs: {args.num_epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"  Total steps per epoch: {len(train_dataloader)}")
    print(f"  Eval every {args.eval_steps} steps")
    print()
    
    # Training loop
    for epoch in range(1, args.num_epochs + 1):
        print(f"\nEpoch {epoch}/{args.num_epochs}")
        print("-" * 40)
        
        epoch_loss = 0.0
        epoch_steps = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                metrics = train_step(
                    model, batch, optimizer, scheduler, args,
                    global_step, accumulation_counter
                )
                
                accumulation_counter += 1
                global_step += 1
                epoch_loss += metrics['loss']
                epoch_steps += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{metrics['loss']:.4f}",
                    'lr': f"{metrics.get('lr', 0):.2e}",
                    'expert': metrics.get('primary_expert', -1)
                })
                
                # Logging
                if global_step % args.logging_steps == 0:
                    avg_loss = epoch_loss / epoch_steps
                    print(f"\n  Step {global_step}: loss={avg_loss:.4f}, "
                          f"router_entropy={metrics.get('router_entropy', 0):.3f}")
                
                # Evaluation
                if global_step % args.eval_steps == 0:
                    eval_results = evaluate(
                        model, test_dataloader, tokenizers, args,
                        num_samples=args.eval_samples
                    )
                    
                    print(f"\n  Evaluation at step {global_step}:")
                    print(f"    BLEU: {eval_results.get('bleu', 0):.2f}")
                    print(f"    chrF: {eval_results.get('chrf', 0):.2f}")
                    print(f"    ROUGE-L: {eval_results.get('rouge_l', 0):.2f}")
                    print(f"    Exact Match: {eval_results.get('exact_match', 0):.2f}%")
                    if 'expert_usage' in eval_results:
                        print(f"    Expert usage: {eval_results['expert_usage']}")
                    
                    # Save best model
                    current_bleu = eval_results.get('bleu', 0)
                    if current_bleu > best_bleu:
                        best_bleu = current_bleu
                        save_checkpoint(
                            model, optimizer, scheduler,
                            global_step, epoch, best_bleu, args,
                            is_best=True
                        )
                    
                    model.train()  # Back to training mode
                
                # Save checkpoint
                if global_step % args.save_steps == 0:
                    save_checkpoint(
                        model, optimizer, scheduler,
                        global_step, epoch, best_bleu, args,
                        is_best=False
                    )
                    
            except Exception as e:
                print(f"\n  Error at step {global_step}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # End of epoch
        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        print(f"\nEpoch {epoch} completed. Average loss: {avg_epoch_loss:.4f}")
    
    print("\n" + "=" * 60)
    print("Training Completed!")
    print(f"Best BLEU score: {best_bleu:.2f}")
    print("=" * 60)
    
    return model


def print_config(args):
    """Print current configuration."""
    print("\n" + "=" * 60)
    print("Current Configuration")
    print("=" * 60)
    print(f"  Data path: {args.data_path}")
    print(f"  Train ratio: {args.train_ratio}")
    print(f"  Test ratio: {args.test_ratio}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Num epochs: {args.num_epochs}")
    print(f"  Max length: {args.max_length}")
    print(f"  Max new tokens: {args.max_new_tokens}")
    print(f"  Num stacks: {args.num_stacks}")
    print(f"  Top-K experts: {args.top_k}")
    print(f"  Use 8-bit: {args.use_8bit}")
    print(f"  Use 4-bit: {args.use_4bit}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Seed: {args.seed}")
    print("=" * 60)


def main():
    """Main entry point."""
    args = parse_args()
    
    # Print configuration
    print_config(args)
    
    # Set random seed
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save args
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Check for data file
    if not os.path.exists(args.data_path):
        print(f"\nError: Data file not found: {args.data_path}")
        sys.exit(1)
    
    # Print dataset statistics
    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)
    stats = get_dataset_statistics(args.data_path)
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v:.1f}" if isinstance(v, float) else f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")
    
    # Load data
    train_data, test_data = load_cpp_cuda_data(
        args.data_path,
        train_ratio=args.train_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    
    # Load expert models
    expert_models, tokenizers = load_expert_models(args)
    
    # Use first tokenizer for data processing (primary tokenizer)
    tokenizer = tokenizers[0]
    
    # Create datasets and dataloaders
    train_dataloader, test_dataloader = create_dataloaders(
        train_data, test_data, tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers
    )
    
    # Initialize MoT model
    model = setup_mot_model(expert_models, tokenizers, args)
    
    # Run evaluation only if specified
    if args.eval_only:
        if args.checkpoint_path:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            load_checkpoint(model, args.checkpoint_path, device)
        
        print("\n" + "=" * 60)
        print("Running Evaluation Only")
        print("=" * 60)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        results = evaluate(model, test_dataloader, tokenizers, args)
        
        evaluator = CudaCodeEvaluator()
        evaluator.print_results(results)
        
        # Save results
        results_path = os.path.join(args.output_dir, 'eval_results.json')
        # Convert any non-serializable items
        save_results = {k: v for k, v in results.items() if isinstance(v, (int, float, str, list, dict))}
        with open(results_path, 'w') as f:
            json.dump(save_results, f, indent=2)
        print(f"Results saved to {results_path}")
        
        return
    
    # Train
    trained_model = train(model, train_dataloader, test_dataloader, tokenizers, args)
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)
    
    final_results = evaluate(trained_model, test_dataloader, tokenizers, args)
    
    evaluator = CudaCodeEvaluator()
    evaluator.print_results(final_results)
    
    # Save final results
    results_path = os.path.join(args.output_dir, 'final_results.json')
    save_results = {k: v for k, v in final_results.items() if isinstance(v, (int, float, str, list, dict))}
    with open(results_path, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"Final results saved to {results_path}")


if __name__ == '__main__':
    main()
