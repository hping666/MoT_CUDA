"""
Simple CUDA code generation script using trained MoT model.

This script provides a function to generate CUDA code from a given prompt
using the Mixture of Thoughts (MoT) framework.

Usage:
    # As a module
    from generate import generate_cuda_code, load_mot_model
    model, tokenizers = load_mot_model("./cuda_mot_output/best_model.pt")
    cuda_code = generate_cuda_code(model, tokenizers, "your prompt here")
    
    # As a script
    python generate.py --prompt "Write a CUDA kernel for vector addition"
    python generate.py --prompt_file input.txt --output_file output.cu
"""

import os
import sys
import argparse
from typing import Optional, Tuple, List, Any

import torch
from transformers import set_seed

from mixture_of_thoughts import MixtureOfThoughts, MoTConfig
from utils import ExpertConfig, ExpertLoader


# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================
DEFAULT_CONFIG = {
    # Model loading settings
    'checkpoint_path': './cuda_mot_output/best_model.pt',
    'use_8bit': False,
    'use_4bit': True,
    'single_gpu': True,
    
    # Generation settings
    'max_new_tokens': 512,
    'temperature': 0.7,
    'top_p': 0.95,
    'use_mot_generate': False,
    
    # Other settings
    'seed': 42,
}

# Default test prompt for demonstration (C++ code to translate to CUDA)
DEFAULT_TEST_PROMPT = """### Translate C++ to CUDA:
void vectorAdd(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}
### CUDA:
"""

# Expert model configuration (must match training)
EXPERT_MODELS = [
    {'name': 'Qwen/Qwen2.5-Coder-1.5B', 'description': 'Qwen2.5 Coder 1.5B'},
    {'name': 'hpcgroup/hpc-coder-v2-1.3b', 'description': 'HPC-Coder-v2 1.3B'},
    {'name': 'bigcode/starcoder2-3b', 'description': 'StarCoder2 3B'},
]


def load_expert_models(
    use_8bit: bool = False,
    use_4bit: bool = True,
    single_gpu: bool = True
) -> Tuple[List[Any], List[Any]]:
    """
    Load expert models for MoT framework.
    
    Args:
        use_8bit: Enable 8-bit quantization
        use_4bit: Enable 4-bit quantization
        single_gpu: Load all models on single GPU
        
    Returns:
        Tuple of (models, tokenizers)
    """
    print("Loading expert models...")
    
    expert_configs = []
    for model_info in EXPERT_MODELS:
        config = ExpertConfig(
            model_name=model_info['name'],
            model_type='causal_lm',
            torch_dtype=torch.float16,
            load_in_8bit=use_8bit,
            load_in_4bit=use_4bit,
            single_gpu=single_gpu,
            target_device='cuda:0',
        )
        expert_configs.append(config)
    
    models, tokenizers = ExpertLoader.load_multiple_experts(expert_configs)
    return models, tokenizers


def load_mot_config_from_checkpoint(checkpoint_path: str, device: torch.device) -> Tuple[dict, dict, dict]:
    """
    Load MoT configuration from checkpoint file.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load checkpoint to
        
    Returns:
        Tuple of (mot_config_dict, saved_args, checkpoint)
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Try new format first (mot_config key)
    mot_config = checkpoint.get('mot_config', None)
    saved_args = checkpoint.get('args', {})
    
    if mot_config is not None:
        return mot_config, saved_args, checkpoint
    
    # Fallback to old format
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


def load_checkpoint_weights(model: MixtureOfThoughts, checkpoint: dict, device: torch.device) -> None:
    """
    Load model weights from checkpoint.
    
    Args:
        model: MoT model instance
        checkpoint: Checkpoint dictionary
        device: Device for loading
    """
    model_state = model.state_dict()
    loaded_keys = 0
    
    for name, param in checkpoint['model_state_dict'].items():
        is_trainable = any(comp in name for comp in ['router', 'interaction', 'auxiliary', 'sentence_encoder'])
        
        if not is_trainable:
            continue
        
        if name in model_state and model_state[name].shape == param.shape:
            model_state[name] = param
            loaded_keys += 1
    
    model.load_state_dict(model_state, strict=False)
    print(f"Loaded {loaded_keys} parameter tensors from checkpoint")


def load_mot_model(
    checkpoint_path: str = DEFAULT_CONFIG['checkpoint_path'],
    use_8bit: bool = DEFAULT_CONFIG['use_8bit'],
    use_4bit: bool = DEFAULT_CONFIG['use_4bit'],
    single_gpu: bool = DEFAULT_CONFIG['single_gpu'],
) -> Tuple[MixtureOfThoughts, List[Any]]:
    """
    Load trained MoT model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        use_8bit: Enable 8-bit quantization
        use_4bit: Enable 4-bit quantization
        single_gpu: Load all models on single GPU
        
    Returns:
        Tuple of (mot_model, tokenizers)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load expert models
    expert_models, tokenizers = load_expert_models(use_8bit, use_4bit, single_gpu)
    
    # Load config from checkpoint
    mot_config_dict, _, checkpoint = load_mot_config_from_checkpoint(checkpoint_path, device)
    
    # Get shared dimension
    hidden_dims = [m.config.hidden_size for m in expert_models]
    shared_dim = mot_config_dict.get('shared_dim', min(hidden_dims))
    
    # Create MoT config
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
    
    # Initialize MoT model
    print("Initializing MoT model...")
    model = MixtureOfThoughts(expert_models, tokenizers, config)
    model = model.to(device)
    
    # Load weights
    load_checkpoint_weights(model, checkpoint, device)
    model.eval()
    
    print("Model loaded successfully!")
    return model, tokenizers


def generate_cuda_code(
    model: MixtureOfThoughts,
    tokenizers: List[Any],
    prompt: str,
    max_new_tokens: int = DEFAULT_CONFIG['max_new_tokens'],
    temperature: float = DEFAULT_CONFIG['temperature'],
    top_p: float = DEFAULT_CONFIG['top_p'],
    use_mot_generate: bool = DEFAULT_CONFIG['use_mot_generate'],
) -> str:
    """
    Generate CUDA code from a given prompt.
    
    Args:
        model: Loaded MoT model
        tokenizers: List of tokenizers for experts
        prompt: Input prompt for CUDA code generation
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Top-p (nucleus) sampling parameter
        use_mot_generate: Use full MoT generation mode (slower but more accurate)
        
    Returns:
        Generated CUDA code string
    """
    model.eval()
    
    with torch.no_grad():
        if use_mot_generate:
            output_ids, primary_idx = model.generate_with_mot(
                raw_text=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        else:
            output_ids, primary_idx = model.generate(
                raw_text=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
    
    # Decode output
    tokenizer = tokenizers[primary_idx]
    prompt_encoding = tokenizer(prompt, return_tensors='pt')
    prompt_len = prompt_encoding['input_ids'].shape[1]
    
    generated_ids = output_ids[0, prompt_len:]
    generated_code = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # Post-process: truncate at stop markers
    for stop_marker in ['\n\n\n', '###', '<|endoftext|>', '<|end|>', '<|im_end|>']:
        if stop_marker in generated_code:
            generated_code = generated_code.split(stop_marker)[0]
    
    return generated_code.strip()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate CUDA code using trained MoT model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input options (mutually exclusive, optional - uses default if not provided)
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument('--prompt', type=str, default=None,
                             help='Direct prompt string')
    input_group.add_argument('--prompt_file', type=str, default=None,
                             help='Path to file containing prompt')
    
    # Output options
    parser.add_argument('--output_file', type=str, default=None,
                        help='Path to save generated code (prints to stdout if not specified)')
    
    # Model options
    parser.add_argument('--checkpoint_path', type=str, default=DEFAULT_CONFIG['checkpoint_path'],
                        help='Path to model checkpoint')
    parser.add_argument('--use_8bit', action='store_true', default=DEFAULT_CONFIG['use_8bit'],
                        help='Enable 8-bit quantization')
    parser.add_argument('--use_4bit', action='store_true', default=DEFAULT_CONFIG['use_4bit'],
                        help='Enable 4-bit quantization')
    parser.add_argument('--single_gpu', action='store_true', default=DEFAULT_CONFIG['single_gpu'],
                        help='Load all models on single GPU')
    
    # Generation options
    parser.add_argument('--max_new_tokens', type=int, default=DEFAULT_CONFIG['max_new_tokens'],
                        help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=DEFAULT_CONFIG['temperature'],
                        help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=DEFAULT_CONFIG['top_p'],
                        help='Top-p sampling parameter')
    parser.add_argument('--use_mot_generate', action='store_true', default=DEFAULT_CONFIG['use_mot_generate'],
                        help='Use full MoT generation mode')
    parser.add_argument('--no_mot_generate', action='store_true',
                        help='Disable MoT generation (use fast mode)')
    
    # Other options
    parser.add_argument('--seed', type=int, default=DEFAULT_CONFIG['seed'],
                        help='Random seed')
    
    return parser.parse_args()


def main():
    """Main entry point for command line usage."""
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Check checkpoint exists
    if not os.path.exists(args.checkpoint_path):
        print(f"Error: Checkpoint not found: {args.checkpoint_path}")
        sys.exit(1)
    
    # Get prompt
    if args.prompt:
        prompt = args.prompt
    elif args.prompt_file:
        if not os.path.exists(args.prompt_file):
            print(f"Error: Prompt file not found: {args.prompt_file}")
            sys.exit(1)
        with open(args.prompt_file, 'r', encoding='utf-8') as f:
            prompt = f.read().strip()
    else:
        # Use default test prompt for demonstration
        print("No prompt provided, using default test prompt for demonstration...")
        prompt = DEFAULT_TEST_PROMPT
    
    # Load model
    model, tokenizers = load_mot_model(
        checkpoint_path=args.checkpoint_path,
        use_8bit=args.use_8bit,
        use_4bit=args.use_4bit,
        single_gpu=args.single_gpu,
    )
    
    # Determine generation mode
    use_mot_generate = args.use_mot_generate and not args.no_mot_generate
    
    print(f"\nGenerating CUDA code...")
    print(f"  Mode: {'full MoT' if use_mot_generate else 'fast'}")
    print(f"  Max new tokens: {args.max_new_tokens}")
    print(f"  Temperature: {args.temperature}")
    print("-" * 50)
    
    # Generate code
    generated_code = generate_cuda_code(
        model=model,
        tokenizers=tokenizers,
        prompt=prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        use_mot_generate=use_mot_generate,
    )
    
    # Output result
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write(generated_code)
        print(f"\nGenerated code saved to: {args.output_file}")
    else:
        print("\n" + "=" * 50)
        print("Generated CUDA Code:")
        print("=" * 50)
        print(generated_code)
        print("=" * 50)


if __name__ == '__main__':
    main()