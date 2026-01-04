"""
Generate code samples using MoT framework for ParEval benchmark evaluation.

This script loads a trained MoT checkpoint and generates code samples for
ParEval prompts. The output is compatible with ParEval's run-all.py evaluator.

Usage:
    python generate_pareval.py \
        --prompts ../../ParEval/prompts/translation-prompts.json \
        --task translation \
        --checkpoint ../cuda_mot_output/best_model.pt \
        --output ./mot_outputs.json \
        --parallelism_model cuda \
        --num_samples 50
"""

import os
import sys
import json
import argparse
import random
import time
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm

import torch
import numpy as np
from transformers import set_seed

# Add parent directory to path for importing MoT modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mixture_of_thoughts import MixtureOfThoughts, MoTConfig
from utils import ExpertConfig, ExpertLoader


# =============================================================================
# Expert Model Configuration
# =============================================================================
# Modify this list to change the expert models used by MoT.
# Each entry should have 'name' (HuggingFace model name) and 'description'.
# The models must match the ones used during MoT training.
# =============================================================================
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


# =============================================================================
# Default Configuration
# =============================================================================
DEFAULT_CONFIG = {
    'task': 'generation',  # 'generation' or 'translation'
    'checkpoint': '../cuda_mot_output/best_model.pt',
    'parallelism_model': 'cuda',
    'prompts': '../../ParEval/prompts/generation-prompts.json',
    'output': './outputs/mot_generation.json',
    'num_samples': 50,
    'temperature': 0.2,
    'top_p': 0.95,
    'max_new_tokens': 1024,
    'batch_size': 1,
    'use_mot_generate': True,
    'use_4bit': True,
    'use_8bit': False,
    'single_gpu': True,
    'seed': 42,
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate code samples using MoT for ParEval evaluation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--prompts', type=str, default=DEFAULT_CONFIG['prompts'],
                        help='Path to ParEval prompts JSON file')
    parser.add_argument('--task', type=str, choices=['generation', 'translation'], default=DEFAULT_CONFIG['task'],
                        help='Task type: generation or translation')
    parser.add_argument('--output', type=str, default=DEFAULT_CONFIG['output'],
                        help='Path to output JSON file')
    
    # Model arguments
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CONFIG['checkpoint'],
                        help='Path to MoT checkpoint')
    
    # Filtering arguments
    parser.add_argument('--parallelism_model', type=str, default=DEFAULT_CONFIG['parallelism_model'],
                        help='Filter prompts by parallelism model (e.g., cuda, omp, mpi)')
    
    # Generation arguments
    parser.add_argument('--num_samples', type=int, default=DEFAULT_CONFIG['num_samples'],
                        help='Number of code samples to generate per prompt')
    parser.add_argument('--temperature', type=float, default=DEFAULT_CONFIG['temperature'],
                        help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=DEFAULT_CONFIG['top_p'],
                        help='Top-p (nucleus) sampling parameter')
    parser.add_argument('--max_new_tokens', type=int, default=DEFAULT_CONFIG['max_new_tokens'],
                        help='Maximum number of new tokens to generate')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_CONFIG['batch_size'],
                        help='Batch size for generation (currently processes one sample at a time)')
    
    # Generation mode
    parser.add_argument('--use_mot_generate', action='store_true', default=DEFAULT_CONFIG['use_mot_generate'],
                        help='Use full MoT generation mode (slower but uses expert interaction)')
    
    # Quantization arguments
    parser.add_argument('--use_4bit', action='store_true', default=DEFAULT_CONFIG['use_4bit'],
                        help='Use 4-bit quantization')
    parser.add_argument('--use_8bit', action='store_true', default=DEFAULT_CONFIG['use_8bit'],
                        help='Use 8-bit quantization')
    parser.add_argument('--no_quantization', action='store_true',
                        help='Disable quantization (use full precision)')
    
    # Other arguments
    parser.add_argument('--single_gpu', action='store_true', default=DEFAULT_CONFIG['single_gpu'],
                        help='Load all models on single GPU')
    parser.add_argument('--seed', type=int, default=DEFAULT_CONFIG['seed'],
                        help='Random seed')
    parser.add_argument('--do_sample', action='store_true', default=True,
                        help='Enable sampling (required for multiple samples per prompt)')
    
    args = parser.parse_args()
    
    # Handle quantization flags
    if args.no_quantization:
        args.use_4bit = False
        args.use_8bit = False
    
    return args


def load_prompts(prompts_path: str, task: str, parallelism_model: str) -> List[Dict[str, Any]]:
    """
    Load and filter ParEval prompts.
    
    Args:
        prompts_path: Path to the prompts JSON file
        task: Task type ('generation' or 'translation')
        parallelism_model: Filter by this parallelism model
        
    Returns:
        List of filtered prompt dictionaries
    """
    print(f"\nLoading prompts from {prompts_path}")
    
    with open(prompts_path, 'r', encoding='utf-8') as f:
        all_prompts = json.load(f)
    
    print(f"  Total prompts loaded: {len(all_prompts)}")
    
    # Filter by parallelism model
    filtered_prompts = [
        p for p in all_prompts 
        if p.get('parallelism_model', '').lower() == parallelism_model.lower()
    ]
    
    print(f"  Prompts with parallelism_model='{parallelism_model}': {len(filtered_prompts)}")
    
    if len(filtered_prompts) == 0:
        # Show available parallelism models
        available_models = set(p.get('parallelism_model', 'unknown') for p in all_prompts)
        print(f"  Available parallelism models: {available_models}")
        raise ValueError(f"No prompts found for parallelism_model='{parallelism_model}'")
    
    # Verify prompt field exists based on task type
    prompt_field = 'translation_prompt' if task == 'translation' else 'prompt'
    valid_prompts = [p for p in filtered_prompts if prompt_field in p]
    
    if len(valid_prompts) < len(filtered_prompts):
        print(f"  Warning: {len(filtered_prompts) - len(valid_prompts)} prompts missing '{prompt_field}' field")
    
    print(f"  Valid prompts for {task} task: {len(valid_prompts)}")
    
    return valid_prompts


def load_expert_models(args) -> Tuple[List[Any], List[Any]]:
    """
    Load expert models for MoT framework.
    
    Args:
        args: Command line arguments
        
    Returns:
        Tuple of (models, tokenizers)
    """
    print("\n" + "=" * 60)
    print("Loading Expert Models")
    print("=" * 60)
    
    # Create ExpertConfig objects from EXPERT_MODELS
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
        print(f"  - {model_info['name']}: {model_info.get('description', '')}")
    
    print()
    models, tokenizers = ExpertLoader.load_multiple_experts(expert_configs)
    return models, tokenizers


def load_mot_config_from_checkpoint(checkpoint_path: str, device: torch.device) -> Tuple[Dict, Dict, Dict]:
    """
    Load MoT configuration from checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the checkpoint to
        
    Returns:
        Tuple of (mot_config_dict, saved_args, checkpoint)
    """
    print(f"\nLoading checkpoint from {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Try new format first (mot_config key)
    mot_config = checkpoint.get('mot_config', None)
    saved_args = checkpoint.get('args', {})
    
    if mot_config is not None:
        print("  Found 'mot_config' in checkpoint")
    else:
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
    
    print(f"  Checkpoint step: {checkpoint.get('step', 'N/A')}")
    print(f"  Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}")
    
    return mot_config, saved_args, checkpoint


def load_checkpoint_weights(model: MixtureOfThoughts, checkpoint: Dict, device: torch.device) -> None:
    """
    Load model weights from checkpoint.
    
    Args:
        model: MoT model instance
        checkpoint: Loaded checkpoint dictionary
        device: Device to load weights to
    """
    print("\nLoading checkpoint weights...")
    
    model_state = model.state_dict()
    loaded_keys = 0
    skipped_keys = 0
    
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
        else:
            skipped_keys += 1
    
    model.load_state_dict(model_state, strict=False)
    
    print(f"  Loaded: {loaded_keys} trainable parameter tensors")
    if skipped_keys > 0:
        print(f"  Skipped: {skipped_keys} keys")


def setup_mot_model(expert_models: List[Any], tokenizers: List[Any], 
                    mot_config_dict: Dict, device: torch.device) -> MixtureOfThoughts:
    """
    Initialize the MoT model with configuration from checkpoint.
    
    Args:
        expert_models: List of expert models
        tokenizers: List of tokenizers
        mot_config_dict: MoT configuration dictionary
        device: Device to place the model on
        
    Returns:
        Initialized MoT model
    """
    print("\n" + "=" * 60)
    print("Initializing MoT Framework")
    print("=" * 60)
    
    # Get hidden dimensions from expert models
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
    print(f"    - Shared dim: {shared_dim}")
    print(f"    - Router hidden dim: {config.router_hidden_dim}")
    print(f"    - Interaction heads: {config.interaction_heads}")
    
    model = MixtureOfThoughts(expert_models, tokenizers, config)
    model = model.to(device)
    
    return model


def get_prompt_text(prompt_dict: Dict, task: str) -> str:
    """
    Extract the prompt text from a prompt dictionary.
    
    Args:
        prompt_dict: Prompt dictionary from ParEval
        task: Task type ('generation' or 'translation')
        
    Returns:
        Prompt text string
    """
    if task == 'translation':
        return prompt_dict['translation_prompt']
    else:
        return prompt_dict['prompt']


def generate_samples_for_prompt(
    model: MixtureOfThoughts,
    prompt_text: str,
    num_samples: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    use_mot_generate: bool,
    do_sample: bool = True
) -> Tuple[List[str], Dict[str, int]]:
    """
    Generate multiple code samples for a single prompt.
    
    Args:
        model: MoT model instance
        prompt_text: Input prompt text
        num_samples: Number of samples to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        max_new_tokens: Maximum new tokens to generate
        use_mot_generate: Whether to use full MoT generation mode
        do_sample: Whether to enable sampling
        
    Returns:
        Tuple of (list of generated codes, expert usage statistics)
    """
    outputs = []
    expert_usage = {}
    
    for _ in range(num_samples):
        try:
            if use_mot_generate:
                # Full MoT generation with expert interaction
                output_ids, primary_idx = model.generate_with_mot(
                    raw_text=prompt_text,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
            else:
                # Fast generation mode
                output_ids, primary_idx = model.generate(
                    raw_text=prompt_text,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                )
            
            # Track expert usage
            expert_usage[primary_idx] = expert_usage.get(primary_idx, 0) + 1
            
            # Decode the generated output
            tokenizer = model.tokenizers[primary_idx]
            
            # Get the full generated text
            full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            # Extract only the newly generated part (after the prompt)
            if full_text.startswith(prompt_text):
                generated_code = full_text[len(prompt_text):]
            else:
                # If prompt is not at the start (due to tokenization differences),
                # try to find and remove it
                prompt_end_markers = ['{\n', '{ \n', '{']
                generated_code = full_text
                for marker in prompt_end_markers:
                    if marker in prompt_text and marker in full_text:
                        # Find position after the last occurrence in prompt
                        prompt_marker_pos = prompt_text.rfind(marker)
                        if prompt_marker_pos != -1:
                            full_marker_pos = full_text.find(marker, prompt_marker_pos)
                            if full_marker_pos != -1:
                                generated_code = full_text[full_marker_pos + len(marker):]
                                break
            
            outputs.append(generated_code.strip())
            
        except Exception as e:
            print(f"    Error generating sample: {e}")
            outputs.append("")  # Add empty string for failed generation
    
    return outputs, expert_usage


def main():
    """Main entry point."""
    args = parse_args()
    
    # Print configuration
    print("\n" + "=" * 60)
    print("ParEval Code Generation with MoT")
    print("=" * 60)
    print(f"  Task: {args.task}")
    print(f"  Prompts file: {args.prompts}")
    print(f"  Output file: {args.output}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Parallelism model filter: {args.parallelism_model}")
    print(f"  Samples per prompt: {args.num_samples}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Top-p: {args.top_p}")
    print(f"  Max new tokens: {args.max_new_tokens}")
    print(f"  Generation mode: {'full MoT' if args.use_mot_generate else 'fast'}")
    print(f"  Quantization: {'4-bit' if args.use_4bit else '8-bit' if args.use_8bit else 'none'}")
    print(f"  Seed: {args.seed}")
    print("=" * 60)
    
    # Set random seeds
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Load prompts
    prompts = load_prompts(args.prompts, args.task, args.parallelism_model)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load expert models
    expert_models, tokenizers = load_expert_models(args)
    
    # Load MoT configuration from checkpoint
    mot_config_dict, saved_args, checkpoint = load_mot_config_from_checkpoint(
        args.checkpoint, device
    )
    
    # Initialize MoT model
    model = setup_mot_model(expert_models, tokenizers, mot_config_dict, device)
    
    # Load checkpoint weights
    load_checkpoint_weights(model, checkpoint, device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Generate code samples
    print("\n" + "=" * 60)
    print("Generating Code Samples")
    print("=" * 60)
    
    results = []
    total_expert_usage = {}
    start_time = time.time()
    
    for prompt_idx, prompt_dict in enumerate(tqdm(prompts, desc="Processing prompts")):
        prompt_text = get_prompt_text(prompt_dict, args.task)
        
        # Generate samples for this prompt
        with torch.no_grad():
            outputs, expert_usage = generate_samples_for_prompt(
                model=model,
                prompt_text=prompt_text,
                num_samples=args.num_samples,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                use_mot_generate=args.use_mot_generate,
                do_sample=args.do_sample,
            )
        
        # Aggregate expert usage
        for expert_idx, count in expert_usage.items():
            total_expert_usage[expert_idx] = total_expert_usage.get(expert_idx, 0) + count
        
        # Build result dictionary preserving original fields
        result = prompt_dict.copy()
        result['outputs'] = outputs
        result['temperature'] = args.temperature
        result['top_p'] = args.top_p
        result['max_new_tokens'] = args.max_new_tokens
        result['do_sample'] = args.do_sample
        result['prompted'] = False  # MoT doesn't use StarCoder-style prompting
        
        results.append(result)
        
        # Periodic progress update
        if (prompt_idx + 1) % 10 == 0:
            elapsed = time.time() - start_time
            rate = (prompt_idx + 1) / elapsed
            print(f"\n  Processed {prompt_idx + 1}/{len(prompts)} prompts "
                  f"({rate:.2f} prompts/min)")
    
    # Save results
    print(f"\nSaving results to {args.output}")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    
    # Print summary
    end_time = time.time()
    total_time = end_time - start_time
    total_samples = len(prompts) * args.num_samples
    
    print("\n" + "=" * 60)
    print("Generation Complete")
    print("=" * 60)
    print(f"  Total prompts processed: {len(prompts)}")
    print(f"  Total samples generated: {total_samples}")
    print(f"  Total time: {total_time:.2f} seconds")
    print(f"  Average time per prompt: {total_time / len(prompts):.2f} seconds")
    print(f"  Average time per sample: {total_time / total_samples:.2f} seconds")
    print(f"\n  Expert usage statistics:")
    for expert_idx, count in sorted(total_expert_usage.items()):
        percentage = count / total_samples * 100
        print(f"    Expert {expert_idx}: {count} samples ({percentage:.1f}%)")
    print(f"\n  Output saved to: {args.output}")
    print("=" * 60)
    
    # Print next steps
    print("\n" + "=" * 60)
    print("Next Steps: Run ParEval Evaluation")
    print("=" * 60)
    print("To evaluate the generated code, run:")
    print(f"\n  cd ~/ParEval/drivers")
    print(f"  python run-all.py {os.path.abspath(args.output)} \\")
    print(f"      -o {os.path.splitext(os.path.abspath(args.output))[0]}_results.json")
    print("\nThen compute metrics:")
    print(f"\n  cd ~/ParEval/analysis")
    print(f"  python metrics.py <results_csv_file> --model-name MoT-CUDA")
    print("=" * 60)


if __name__ == '__main__':
    main()