# Mixture of Thoughts (MoT) for CUDA Code Generation

A framework for orchestrating multiple Large Language Models (LLMs) using sparse top-k routing, specialized for **C++ to CUDA code translation**. MoT dynamically selects and combines outputs from multiple expert code models based on input characteristics.

## Overview

This implementation applies the Mixture of Thoughts framework to the task of translating C++ code into CUDA kernels. The system uses three specialized code generation models as experts:

- **Qwen2.5-Coder**: Strong general code generation capabilities
- **HPC-Coder-v2**: Specialized for HPC and parallel code patterns  
- **StarCoder2**: Multi-language code generation with broad coverage

A learned router dynamically selects the most appropriate experts for each input, combining their outputs through cross-attention interaction layers and stack-based layer partitioning.

### Key Features

- **Sparse Top-K Routing**: Gumbel-Softmax based differentiable expert selection
- **Stack-Based Layer Partitioning**: Divides model layers into stacks for fine-grained expert interaction
- **Cross-Expert Attention**: Enables information flow between selected experts at each stack
- **Routing Consistency Loss**: Encourages stable outputs under different routing perturbations
- **4-bit/8-bit Quantization**: Memory-efficient expert model loading via bitsandbytes
- **Comprehensive Evaluation**: BLEU, chrF, ROUGE-L, Exact Match, Edit Similarity, and Syntax Validity metrics

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    MixtureOfThoughts Framework                  │
├─────────────────────────────────────────────────────────────────┤
│  Input: C++ Source Code                                         │
│       ↓                                                         │
│  ┌──────────────────┐                                          │
│  │ Sentence Encoder │  (DeBERTa-v3-large)                      │
│  │  Prompt → Vector │                                          │
│  └────────┬─────────┘                                          │
│           ↓                                                     │
│  ┌──────────────────┐                                          │
│  │  Sparse Router   │  Gumbel-Softmax + Top-K Selection        │
│  │  (Trainable MLP) │                                          │
│  └────────┬─────────┘                                          │
│           ↓                                                     │
│  ┌──────────────────────────────────────────┐                  │
│  │         Expert Models (Frozen)           │                  │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐   │                  │
│  │  │  Qwen   │ │   HPC   │ │  Star   │   │                  │
│  │  │  Coder  │ │  Coder  │ │ Coder2  │   │                  │
│  │  └────┬────┘ └────┬────┘ └────┬────┘   │                  │
│  └───────┼───────────┼───────────┼────────┘                  │
│          ↓           ↓           ↓                             │
│  ┌──────────────────────────────────────────┐                  │
│  │      Interaction Layers (Trainable)      │                  │
│  │   Stack 1 → Stack 2 → ... → Stack N      │                  │
│  │   (Cross-Expert Attention Mechanism)     │                  │
│  └──────────────────────────────────────────┘                  │
│          ↓                                                     │
│  ┌──────────────────┐                                          │
│  │  Primary Expert  │                                          │
│  │    LM Head       │  → Output: CUDA Kernel Code              │
│  └──────────────────┘                                          │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

### Requirements

- Python 3.8 or higher
- CUDA-capable GPU with at least 16GB VRAM (24GB+ recommended)
- PyTorch 2.0.0 or higher

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd mot-cuda
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Required packages:
```
torch>=2.0.0
transformers>=4.35.0
sentence-transformers>=2.2.0
bitsandbytes>=0.41.0
accelerate>=0.24.0
tqdm
numpy
```

## Project Structure

```
mot-cuda/
├── mixture_of_thoughts.py    # Core MoT framework implementation
├── train_cuda_mot.py         # Training script for C++ to CUDA translation
├── evaluate.py               # Standalone evaluation script
├── cuda_dataset.py           # Dataset loading and preprocessing
├── cuda_evaluation.py        # Evaluation metrics (BLEU, chrF, etc.)
├── utils.py                  # Helper functions and model loading utilities
├── cpp_cuda_train_synthetic.jsonl  # Training data (JSONL format)
└── cuda_mot_output/          # Output directory for checkpoints and results
```

## Data Format

The training data should be in JSONL format with the following structure:

```json
{"cpp": "void add(int* a, int* b, int n) { for(int i=0; i<n; i++) a[i] += b[i]; }", "generated_cuda": "__global__ void add(int* a, int* b, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if(i < n) a[i] += b[i]; }"}
```

Each line contains:
- `cpp`: The source C++ code to translate
- `generated_cuda`: The target CUDA kernel code

## Usage

### Quick Start - Training

```bash
python train_cuda_mot.py \
    --data_path ./cpp_cuda_train_synthetic.jsonl \
    --use_4bit \
    --num_epochs 3 \
    --output_dir ./cuda_mot_output
```

### Quick Start - Evaluation

```bash
python evaluate.py \
    --checkpoint_path ./cuda_mot_output/best_model.pt \
    --data_path ./cpp_cuda_train_synthetic.jsonl \
    --use_4bit
```

### Training Configuration

Key training parameters:

```bash
python train_cuda_mot.py \
    --data_path ./cpp_cuda_train_synthetic.jsonl \
    --train_ratio 0.8 \
    --test_ratio 0.1 \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_epochs 3 \
    --learning_rate 5e-6 \
    --max_length 512 \
    --max_new_tokens 256 \
    --num_stacks 4 \
    --top_k 2 \
    --lambda_consistency 0.05 \
    --best_model_metric bleu \
    --use_4bit \
    --single_gpu \
    --output_dir ./cuda_mot_output
```

### Evaluation Configuration

```bash
python evaluate.py \
    --checkpoint_path ./cuda_mot_output/best_model.pt \
    --data_path ./cpp_cuda_train_synthetic.jsonl \
    --test_ratio 0.1 \
    --seed 42 \
    --max_new_tokens 256 \
    --use_4bit \
    --single_gpu \
    --output_dir ./eval_output \
    --save_predictions
```

## Core Parameters

### MoT Architecture Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_stacks` | 4 | Number of layer stacks to partition each expert model into |
| `top_k` | 2 | Number of experts to activate for each input |
| `shared_dim` | min(expert_dims) | Shared dimension for cross-expert interaction |
| `router_hidden_dim` | 512 | Hidden dimension of the router MLP |
| `interaction_heads` | 8 | Number of attention heads in interaction layers |

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 5e-6 | Base learning rate (router uses 2x, auxiliary uses 0.5x) |
| `batch_size` | 1 | Training batch size per step |
| `gradient_accumulation_steps` | 8 | Steps to accumulate before optimizer update |
| `gradient_clip` | 0.3 | Maximum gradient norm for clipping |
| `lambda_consistency` | 0.05 | Weight for routing consistency loss |
| `warmup_steps` | 200 | Number of warmup steps for scheduler |

### Quantization Options

| Parameter | Description |
|-----------|-------------|
| `--use_4bit` | Load expert models in 4-bit quantization (recommended) |
| `--use_8bit` | Load expert models in 8-bit quantization |
| `--single_gpu` | Load all models on single GPU (required for layer-by-layer forward) |

### Best Model Selection

| Metric | Description |
|--------|-------------|
| `bleu` | BLEU-4 score (default) |
| `chrf` | Character-level F-score |
| `rouge_l` | ROUGE-L F1 score |
| `exact_match` | Exact match percentage |
| `edit_similarity` | Edit distance similarity |

## Loss Function

The total training loss combines multiple components:

```
L_total = L_lm + 0.01 * L_entropy + 0.01 * L_balance + λ * L_consistency
```

| Component | Description |
|-----------|-------------|
| `L_lm` | Language modeling cross-entropy loss (only on CUDA output tokens) |
| `L_entropy` | Router entropy regularization (encourages exploration) |
| `L_balance` | Load balancing loss (prevents expert collapse) |
| `L_consistency` | Routing consistency loss (symmetric KL divergence between two forward passes with different Gumbel noise) |

## Generation Modes

### Fast Mode (Default)
```bash
python evaluate.py --checkpoint_path ./best_model.pt
```
- Routes once at the beginning
- Primary expert generates independently using `model.generate()`
- Much faster inference
- Best for production use

### Full MoT Mode
```bash
python evaluate.py --checkpoint_path ./best_model.pt --use_mot_generate
```
- Each token generated through full MoT forward pass
- Expert interaction at every step
- Slower but potentially higher quality
- Best for research and comparison

## Evaluation Metrics

The framework computes comprehensive code generation metrics:

| Metric | Range | Description |
|--------|-------|-------------|
| **BLEU-4** | 0-100 | N-gram precision with brevity penalty |
| **chrF** | 0-100 | Character-level F-score |
| **ROUGE-L** | 0-100 | Longest common subsequence F1 |
| **Exact Match** | 0-100% | Percentage of exact matches |
| **Edit Similarity** | 0-100% | 1 - normalized Levenshtein distance |
| **Syntax Validity** | 0-100% | CUDA syntax validity check |

## Example Code

### Basic Usage

```python
from mixture_of_thoughts import MixtureOfThoughts, MoTConfig
from utils import ExpertConfig, ExpertLoader
import torch

# Define expert configurations
expert_configs = [
    ExpertConfig(
        model_name='Qwen/Qwen2.5-Coder-1.5B',
        model_type='causal_lm',
        load_in_4bit=True,
        single_gpu=True,
        target_device='cuda:0'
    ),
    ExpertConfig(
        model_name='hpcgroup/hpc-coder-v2-1.3b',
        model_type='causal_lm',
        load_in_4bit=True,
        single_gpu=True,
        target_device='cuda:0'
    ),
    ExpertConfig(
        model_name='bigcode/starcoder2-3b',
        model_type='causal_lm',
        load_in_4bit=True,
        single_gpu=True,
        target_device='cuda:0'
    ),
]

# Load experts
models, tokenizers = ExpertLoader.load_multiple_experts(expert_configs)

# Configure MoT
config = MoTConfig(
    num_stacks=4,
    top_k=2,
    shared_dim=min(m.config.hidden_size for m in models),
    lambda_consistency=0.05
)

# Initialize framework
mot_model = MixtureOfThoughts(
    expert_models=models,
    tokenizers=tokenizers,
    config=config
)
mot_model = mot_model.to('cuda')

# Translate C++ to CUDA
cpp_code = """
void vectorAdd(float* a, float* b, float* c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}
"""

cuda_code, expert_idx = mot_model.translate(
    source_text=cpp_code,
    prompt_template="### Translate C++ to CUDA:\n{source}\n### CUDA:\n",
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.95
)

print(f"Generated by expert {expert_idx}:")
print(cuda_code)
```

### Getting Routing Information

```python
# Analyze how the router scores different experts
routing_info = mot_model.get_routing_info(cpp_code)
print(f"Primary expert: {routing_info['primary_expert']}")
print(f"Active experts: {routing_info['active_experts']}")
print(f"Expert probabilities: {routing_info['all_expert_probs']}")
```

## Output Structure

After training, the output directory contains:

```
cuda_mot_output/
├── args.json                      # Training configuration
├── best_model.pt                  # Best checkpoint (by selected metric)
├── checkpoint-{step}.pt           # Periodic checkpoints
├── final_results_last_epoch.json  # Evaluation with last epoch weights
└── final_results_best_model.json  # Evaluation with best model weights
```

Checkpoint contents:
```python
{
    'step': int,
    'epoch': int,
    'model_state_dict': dict,      # Router + Interaction layers only
    'optimizer_state_dict': dict,
    'scheduler_state_dict': dict,
    'best_metric': float,
    'best_metric_name': str,
    'args': dict
}
```

## Changing Expert Models

To use different expert models, modify the `EXPERT_MODELS` list in `train_cuda_mot.py`:

```python
# For larger models (requires more VRAM)
EXPERT_MODELS = [
    {
        'name': 'Qwen/Qwen2.5-Coder-7B',
        'description': 'Qwen2.5 Coder 7B - Strong general code generation',
    },
    {
        'name': 'hpcgroup/hpc-coder-v2-6.7b',
        'description': 'HPC-Coder-v2 - Specialized for HPC/parallel code',
    },
    {
        'name': 'bigcode/starcoder2-7b',
        'description': 'StarCoder2 7B - Multi-language code generation',
    },
]
```

## Performance Considerations

- **Memory**: 4-bit quantization reduces VRAM usage by ~4x
- **Speed**: Use fast generation mode for inference (default)
- **Batch Size**: Keep at 1 with gradient accumulation for stability
- **Gradient Clipping**: Essential for training stability with quantized models

### Recommended Hardware

| Configuration | Minimum VRAM | Recommended |
|--------------|--------------|-------------|
| 1.5B experts (4-bit) | 12GB | 16GB |
| 3B experts (4-bit) | 16GB | 24GB |
| 7B experts (4-bit) | 24GB | 48GB |

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce `max_length` (e.g., 256 instead of 512)
   - Ensure `--use_4bit` is enabled
   - Reduce `gradient_accumulation_steps`

2. **NaN Loss**
   - Check if `gradient_clip` is set (default: 0.3)
   - Reduce learning rate
   - The code includes automatic NaN detection and skipping

3. **Slow Training**
   - Ensure `--single_gpu` is set
   - Use smaller expert models for experimentation
   - Disable `--use_mot_generate` during evaluation

4. **Model Download Failures**
   - Check HuggingFace hub access
   - Models are cached in `~/.cache/huggingface/hub`

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check GPU memory:
```python
from utils import print_gpu_memory
print_gpu_memory()
```

## Technical Details

### Numerical Stability

The framework implements several stability measures:
- InteractionLayer uses float32 internally
- Hidden states clamped to [-100, 100]
- LayerNorm applied before and after projections
- Automatic NaN/Inf detection and replacement
- Gradient clipping with configurable threshold

### Differentiable Routing

- **Gumbel-Softmax**: Enables gradient flow through discrete expert selection
- **Straight-Through Estimator**: Hard selection forward, soft gradients backward
- **Consistency Loss**: Symmetric KL divergence encourages stable routing

### What Gets Trained

Only these components have `requires_grad=True`:
- Router MLP (~1M parameters)
- Interaction Layers (~10M parameters per stack)
- Sentence Encoder (if not frozen)

Expert models are completely frozen to preserve their pre-trained capabilities.


## License

This project is licensed under the MIT License. See LICENSE file for details.


