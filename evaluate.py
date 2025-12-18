"""
Standalone evaluation/inference script for Mixture of Thoughts (MoT) model.

This script loads a trained MoT checkpoint and evaluates it on the test set.
The test set is determined by the same seed and test_ratio used during training.
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

# Local imports
from mixture_of_thoughts import MixtureOfThoughts, MoTConfig
from cuda_dataset import load_cpp_cuda_data, create_dataloaders, get_dataset_statistics
from cuda_evaluation import CudaCodeEvaluator
from utils import ExpertConfig, ExpertLoader, compute_model_size


# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================
DEFAULT_CONFIG = {
    # Data settings
    'data_path': './cpp_cuda_train_synthetic.jsonl',
    'test_ratio': 0.1,
    'seed': 42,
    
    # Model settings
    'checkpoint_path': './cuda_mot_output/best_model.pt',
    'use_8bit': False,
    'use_4bit': True,
    'single_gpu': True,
    
    # Evaluation settings
    'max_new_tokens': 256,
    'use_mot_generate': False,
    'num_samples': None,  # None means evaluate all test samples
    'batch_size': 1,
    'prompt_template': "### Translate C++ to CUDA:\n{source}\n### CUDA:\n",  # Template for code translation
    
    # Output settings
    'output_dir': './eval_output',
    'save_predictions': True,
}

# =============================================================================
# EXPERT MODEL CONFIGURATION (must match training)
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
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate MoT model on test set',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default=DEFAULT_CONFIG['data_path'],
                        help='Path to the JSONL dataset file')
    parser.add_argument('--test_ratio', type=float, default=DEFAULT_CONFIG['test_ratio'],
                        help='Ratio of data to use for testing')
    parser.add_argument('--seed', type=int, default=DEFAULT_CONFIG['seed'],
                        help='Random seed (must match training for consistent test split)')
    
    # Model arguments
    parser.add_argument('--checkpoint_path', type=str, default=DEFAULT_CONFIG['checkpoint_path'],
                        help='Path to model checkpoint (default: best_model.pt)')
    parser.add_argument('--use_8bit', action='store_true', default=DEFAULT_CONFIG['use_8bit'],
                        help='Load models in 8-bit quantization')
    parser.add_argument('--use_4bit', action='store_true', default=DEFAULT_CONFIG['use_4bit'],
                        help='Load models in 4-bit quantization')
    parser.add_argument('--single_gpu', action='store_true', default=DEFAULT_CONFIG['single_gpu'],
                        help='Load all models on single GPU')
    
    # Evaluation arguments
    parser.add_argument('--max_new_tokens', type=int, default=DEFAULT_CONFIG['max_new_tokens'],
                        help='Maximum new tokens to generate')
    parser.add_argument('--use_mot_generate', action='store_true', default=DEFAULT_CONFIG['use_mot_generate'],
                        help='Use full MoT generation (slow but uses expert interaction)')
    parser.add_argument('--num_samples', type=int, default=DEFAULT_CONFIG['num_samples'],
                        help='Number of samples to evaluate (None = all)')
    parser.add_argument('--prompt_template', type=str, default=DEFAULT_CONFIG['prompt_template'],
                        help='Prompt template for code translation. Use {source} as placeholder for input code')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default=DEFAULT_CONFIG['output_dir'],
                        help='Directory to save evaluation results')
    parser.add_argument('--save_predictions', action='store_true', default=DEFAULT_CONFIG['save_predictions'],
                        help='Save individual predictions to file')
    
    return parser.parse_args()


def load_expert_models(args) -> Tuple[List[Any], List[Any]]:
    """Load expert models."""
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


def load_checkpoint(model, checkpoint_path, device):
    """Load model checkpoint."""
    print(f"\nLoading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load weights
    model_state = model.state_dict()
    loaded_keys = 0
    for name, param in checkpoint['model_state_dict'].items():
        if name in model_state:
            model_state[name] = param
            loaded_keys += 1
    model.load_state_dict(model_state)
    
    # Print checkpoint info
    print(f"  Loaded {loaded_keys} parameter tensors")
    print(f"  Checkpoint step: {checkpoint.get('step', 'N/A')}")
    print(f"  Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}")
    if 'best_metric_name' in checkpoint:
        print(f"  Best metric: {checkpoint['best_metric_name']} = {checkpoint.get('best_metric', 'N/A'):.2f}")
    
    return checkpoint


@torch.no_grad()
def evaluate(model, test_dataloader, args) -> Tuple[Dict[str, float], List[Dict]]:
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
            
            # Store individual result
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
    
    # Compute metrics
    if predictions:
        metrics = evaluator.evaluate_all(predictions, references)
        metrics['expert_usage'] = expert_usage
        metrics['generation_mode'] = 'mot' if args.use_mot_generate else 'fast'
    else:
        metrics = {'error': 'No predictions generated'}
    
    return metrics, all_results


def main():
    """Main entry point."""
    args = parse_args()
    
    # Print configuration
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
    print(f"  Prompt template: {args.prompt_template[:50]}...")
    print(f"  Output dir: {args.output_dir}")
    print("=" * 60)
    
    # Set seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check files exist
    if not os.path.exists(args.data_path):
        print(f"\nError: Data file not found: {args.data_path}")
        sys.exit(1)
    if not os.path.exists(args.checkpoint_path):
        print(f"\nError: Checkpoint not found: {args.checkpoint_path}")
        sys.exit(1)
    
    # Load data (use same split as training)
    print("\n" + "=" * 60)
    print("Loading Dataset")
    print("=" * 60)
    _, test_data = load_cpp_cuda_data(
        args.data_path,
        train_ratio=1.0 - args.test_ratio,  # Ensure same split
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    print(f"  Test samples: {len(test_data)}")
    
    # Load expert models
    expert_models, tokenizers = load_expert_models(args)
    
    # Load checkpoint to get MoT config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    saved_args = checkpoint.get('args', {})
    
    # Initialize MoT model
    print("\n" + "=" * 60)
    print("Initializing MoT Framework")
    print("=" * 60)
    
    hidden_dims = [m.config.hidden_size for m in expert_models]
    config = MoTConfig(
        num_stacks=saved_args.get('num_stacks', 4),
        top_k=saved_args.get('top_k', 2),
        shared_dim=min(hidden_dims),
        lambda_consistency=saved_args.get('lambda_consistency', 0.05),
    )
    print(f"  Num stacks: {config.num_stacks}")
    print(f"  Top-K: {config.top_k}")
    
    model = MixtureOfThoughts(expert_models, tokenizers, config)
    model = model.to(device)
    
    # Load checkpoint weights
    load_checkpoint(model, args.checkpoint_path, device)
    
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
