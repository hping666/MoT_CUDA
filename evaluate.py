"""
Standalone evaluation/inference script for Mixture of Thoughts (MoT) model.

This script loads a trained MoT checkpoint and evaluates it on the test set.
MoT configuration is read entirely from the checkpoint file.
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Optional, Tuple, Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import set_seed

from mixture_of_thoughts import MixtureOfThoughts, MoTConfig
from cuda_dataset import load_cpp_cuda_data, create_dataloaders, get_dataset_statistics
from cuda_evaluation import CudaCodeEvaluator
from utils import ExpertConfig, ExpertLoader, compute_model_size


DEFAULT_CONFIG = {
    'data_path': './cpp_cuda_train_synthetic.jsonl',
    'test_ratio': 0.1,
    'seed': 42,
    'checkpoint_path': './cuda_mot_output/best_model.pt',
    'use_8bit': False,
    'use_4bit': True,
    'single_gpu': True,
    'max_new_tokens': 256,
    'use_mot_generate': True,
    'num_samples': None,    # None means use all samples
    'prompt_template': "### Translate C++ to CUDA:\n{source}\n### CUDA:\n",
    'output_dir': './eval_output',
    'save_predictions': True,
}

EXPERT_MODELS = [
    {'name': 'Qwen/Qwen2.5-Coder-1.5B', 'description': 'Qwen2.5 Coder 1.5B'},
    {'name': 'hpcgroup/hpc-coder-v2-1.3b', 'description': 'HPC-Coder-v2 1.3B'},
    {'name': 'bigcode/starcoder2-3b', 'description': 'StarCoder2 3B'},
]


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate MoT model on test set')
    parser.add_argument('--data_path', type=str, default=DEFAULT_CONFIG['data_path'])
    parser.add_argument('--test_ratio', type=float, default=DEFAULT_CONFIG['test_ratio'])
    parser.add_argument('--seed', type=int, default=DEFAULT_CONFIG['seed'])
    parser.add_argument('--checkpoint_path', type=str, default=DEFAULT_CONFIG['checkpoint_path'])
    parser.add_argument('--use_8bit', action='store_true', default=DEFAULT_CONFIG['use_8bit'])
    parser.add_argument('--use_4bit', action='store_true', default=DEFAULT_CONFIG['use_4bit'])
    parser.add_argument('--single_gpu', action='store_true', default=DEFAULT_CONFIG['single_gpu'])
    parser.add_argument('--max_new_tokens', type=int, default=DEFAULT_CONFIG['max_new_tokens'])
    parser.add_argument('--use_mot_generate', action='store_true', default=DEFAULT_CONFIG['use_mot_generate'])
    parser.add_argument('--num_samples', type=int, default=DEFAULT_CONFIG['num_samples'])
    parser.add_argument('--prompt_template', type=str, default=DEFAULT_CONFIG['prompt_template'])
    parser.add_argument('--output_dir', type=str, default=DEFAULT_CONFIG['output_dir'])
    parser.add_argument('--save_predictions', action='store_true', default=DEFAULT_CONFIG['save_predictions'])
    return parser.parse_args()


def load_expert_models(args):
    print("\n" + "=" * 60)
    print("Loading Expert Models")
    print("=" * 60)
    
    expert_configs = []
    for model_info in EXPERT_MODELS:
        config = ExpertConfig(
            model_name=model_info['name'],
            model_type='causal_lm',
            torch_dtype=torch.float16,
            load_in_8bit=args.use_8bit,
            load_in_4bit=args.use_4bit,
            single_gpu=args.single_gpu,
            target_device='cuda:0',
        )
        expert_configs.append(config)
        print(f"  - {model_info['name']}")
    
    models, tokenizers = ExpertLoader.load_multiple_experts(expert_configs)
    return models, tokenizers


def load_mot_config_from_checkpoint(checkpoint_path, device):
    """Load MoT configuration from checkpoint."""
    print(f"\nReading configuration from checkpoint...")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Try new format first (mot_config key)
    mot_config = checkpoint.get('mot_config', None)
    saved_args = checkpoint.get('args', {})
    
    if mot_config is not None:
        print("  Found 'mot_config' in checkpoint")
        return mot_config, saved_args, checkpoint
    
    # Fallback to old format (reconstruct from args)
    print("  Reconstructing config from 'args' (old checkpoint format)")
    mot_config = {
        'num_stacks': saved_args.get('num_stacks', 4),
        'top_k': saved_args.get('top_k', 2),
        'router_hidden_dim': saved_args.get('router_hidden_dim', 512),
        'router_temperature': saved_args.get('router_temperature', 1.0),
        'interaction_heads': saved_args.get('interaction_heads', 8),
        'dropout_rate': saved_args.get('dropout_rate', 0.1),
        'use_gumbel': saved_args.get('use_gumbel', True),
        'gumbel_temperature': saved_args.get('gumbel_temperature', 1.0),
        'lambda_consistency': saved_args.get('lambda_consistency', 0.05),
        'consistency_temperature': saved_args.get('consistency_temperature', 2.0),
    }
    
    return mot_config, saved_args, checkpoint


def load_checkpoint_weights(model, checkpoint, device):
    """Load model weights from checkpoint."""
    print("\nLoading checkpoint weights...")
    
    model_state = model.state_dict()
    loaded_keys = 0
    skipped_keys = 0
    mismatched_keys = 0
    
    for name, param in checkpoint['model_state_dict'].items():
        # Only load trainable components
        is_trainable = any(comp in name for comp in ['router', 'interaction', 'auxiliary', 'sentence_encoder'])
        
        if not is_trainable:
            skipped_keys += 1
            continue
        
        if name in model_state:
            if model_state[name].shape == param.shape:
                model_state[name] = param
                loaded_keys += 1
            else:
                print(f"  Shape mismatch: {name}")
                mismatched_keys += 1
        else:
            skipped_keys += 1
    
    model.load_state_dict(model_state, strict=False)
    
    print(f"  Loaded: {loaded_keys} tensors")
    if skipped_keys > 0:
        print(f"  Skipped: {skipped_keys} keys")
    if mismatched_keys > 0:
        print(f"  Mismatched: {mismatched_keys} keys")
    
    print(f"  Checkpoint step: {checkpoint.get('step', 'N/A')}")
    print(f"  Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}")
    
    best_metric = checkpoint.get('best_metric', None)
    best_metric_name = checkpoint.get('best_metric_name', None)
    if best_metric is not None and best_metric_name is not None:
        print(f"  Best metric: {best_metric_name} = {best_metric:.2f}")


@torch.no_grad()
def evaluate(model, test_dataloader, args):
    """Evaluate model on test set."""
    model.eval()
    
    predictions = []
    references = []
    all_results = []
    expert_usage = {}
    
    evaluator = CudaCodeEvaluator()
    max_samples = args.num_samples or len(test_dataloader)
    mode_str = "full MoT" if args.use_mot_generate else "fast"
    
    print(f"\nEvaluating on {min(max_samples, len(test_dataloader))} samples")
    print(f"  Generation mode: {mode_str}")
    print(f"  Max new tokens: {args.max_new_tokens}")
    
    for i, batch in enumerate(tqdm(test_dataloader, desc="Evaluating", total=min(max_samples, len(test_dataloader)))):
        if i >= max_samples:
            break
        
        cpp_code = batch['cpp_code'][0]
        reference_cuda = batch['cuda_code'][0]
        
        try:
            generated_cuda, primary_idx = model.translate(
                source_text=cpp_code,
                prompt_template=args.prompt_template,
                max_new_tokens=args.max_new_tokens,
                temperature=0.7,
                top_p=0.95,
                use_mot_generate=args.use_mot_generate,
            )
            
            expert_usage[primary_idx] = expert_usage.get(primary_idx, 0) + 1
            predictions.append(generated_cuda)
            references.append(reference_cuda)
            
            all_results.append({
                'index': i,
                'cpp_code': cpp_code,
                'reference_cuda': reference_cuda,
                'generated_cuda': generated_cuda,
                'primary_expert': primary_idx,
            })
            
        except Exception as e:
            print(f"\n  Error at sample {i}: {e}")
            predictions.append("")
            references.append(reference_cuda)
            all_results.append({
                'index': i,
                'cpp_code': cpp_code,
                'reference_cuda': reference_cuda,
                'generated_cuda': "",
                'error': str(e),
            })
    
    if predictions:
        metrics = evaluator.evaluate_all(predictions, references)
        metrics['expert_usage'] = expert_usage
        metrics['generation_mode'] = 'mot' if args.use_mot_generate else 'fast'
    else:
        metrics = {'error': 'No predictions generated'}
    
    return metrics, all_results


def main():
    args = parse_args()
    
    print("\n" + "=" * 60)
    print("Evaluation Configuration")
    print("=" * 60)
    print(f"  Data path: {args.data_path}")
    print(f"  Test ratio: {args.test_ratio}")
    print(f"  Seed: {args.seed}")
    print(f"  Checkpoint: {args.checkpoint_path}")
    print(f"  Max new tokens: {args.max_new_tokens}")
    print(f"  Use MoT generate: {args.use_mot_generate}")
    print(f"  Num samples: {args.num_samples or 'all'}")
    print(f"  Output dir: {args.output_dir}")
    print("=" * 60)
    
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    if not os.path.exists(args.data_path):
        print(f"\nError: Data file not found: {args.data_path}")
        sys.exit(1)
    if not os.path.exists(args.checkpoint_path):
        print(f"\nError: Checkpoint not found: {args.checkpoint_path}")
        sys.exit(1)
    
    # Load dataset
    print("\n" + "=" * 60)
    print("Loading Dataset")
    print("=" * 60)
    _, test_data = load_cpp_cuda_data(
        args.data_path,
        train_ratio=1.0 - args.test_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    print(f"  Test samples: {len(test_data)}")
    
    # Load expert models
    expert_models, tokenizers = load_expert_models(args)
    
    # Load MoT config from checkpoint
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mot_config_dict, saved_args, checkpoint = load_mot_config_from_checkpoint(args.checkpoint_path, device)
    
    # Initialize MoT model with config from checkpoint
    print("\n" + "=" * 60)
    print("Initializing MoT Framework")
    print("=" * 60)
    
    hidden_dims = [m.config.hidden_size for m in expert_models]
    shared_dim = mot_config_dict.get('shared_dim', min(hidden_dims))
    
    config = MoTConfig(
        num_stacks=mot_config_dict['num_stacks'],
        top_k=mot_config_dict['top_k'],
        shared_dim=shared_dim,
        router_hidden_dim=mot_config_dict['router_hidden_dim'],
        router_temperature=mot_config_dict.get('router_temperature', 1.0),
        interaction_heads=mot_config_dict['interaction_heads'],
        dropout_rate=mot_config_dict.get('dropout_rate', 0.1),
        use_gumbel=mot_config_dict.get('use_gumbel', True),
        gumbel_temperature=mot_config_dict.get('gumbel_temperature', 1.0),
        lambda_consistency=mot_config_dict.get('lambda_consistency', 0.05),
        consistency_temperature=mot_config_dict.get('consistency_temperature', 2.0),
    )
    
    print(f"  Config from checkpoint:")
    print(f"    - Num stacks: {config.num_stacks}")
    print(f"    - Top-K: {config.top_k}")
    print(f"    - Router hidden dim: {config.router_hidden_dim}")
    print(f"    - Interaction heads: {config.interaction_heads}")
    
    model = MixtureOfThoughts(expert_models, tokenizers, config)
    model = model.to(device)
    
    # Load checkpoint weights
    load_checkpoint_weights(model, checkpoint, device)
    
    # Create dataloader
    _, test_dataloader = create_dataloaders(
        [], test_data, tokenizers[0],
        batch_size=1,
        max_length=saved_args.get('max_length', 512),
    )
    
    # Run evaluation
    print("\n" + "=" * 60)
    print("Running Evaluation")
    print("=" * 60)
    
    metrics, all_results = evaluate(model, test_dataloader, args)
    
    # Print results
    evaluator = CudaCodeEvaluator()
    evaluator.print_results(metrics)
    
    # Save metrics
    metrics_path = os.path.join(args.output_dir, 'eval_metrics.json')
    save_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float, str, list, dict))}
    save_metrics['checkpoint_path'] = args.checkpoint_path
    save_metrics['num_samples_evaluated'] = len(all_results)
    with open(metrics_path, 'w') as f:
        json.dump(save_metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")
    
    # Save predictions
    if args.save_predictions:
        predictions_path = os.path.join(args.output_dir, 'predictions.json')
        with open(predictions_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"Predictions saved to {predictions_path}")
    
    print("\n" + "=" * 60)
    print("Evaluation Complete")
    print("=" * 60)


if __name__ == '__main__':
    main()