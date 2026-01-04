# Mixture of Thoughts (MoT)

A framework for orchestrating multiple Large Language Models (LLMs) using sparse top-k routing. MoT dynamically selects and combines outputs from multiple expert models based on input characteristics, enabling efficient and specialized model deployment.

## Overview

The Mixture of Thoughts framework implements a multi-expert system where different language models act as specialized experts. A learned router dynamically selects the most appropriate experts for each input, combining their outputs through attention mechanisms and stack-based layer partitioning.

### Key Features

- We use Sparse Top-K Routing. You can plug and play any router as you please.
- **Stack-Based Layer Partitioning**: Divides model layers into stacks for fine-grained expert interaction
- **Cross-Expert Attention**: Enables information flow between selected experts
- **Distributed Training**: Supports multi-GPU training with DDP (Distributed Data Parallel)
- **Multiple Benchmark Support**: Compatible with MMLU, GSM8K, CMMLU, ARC-Challenge, HumanEval, and ParEval datasets

## Installation

### Requirements

- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- PyTorch 2.0.0 or higher

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd mot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download expert models (optional, will download automatically during first use):
```bash
python download_experts.py --config configs/routerdc_mot_config.json
```

## Project Structure

```
mot/
├── mixture_of_thoughts.py    # Core MoT framework implementation
├── training.py               # Training utilities and loss functions
├── train_ddp.py             # Distributed training script
├── dataset_loaders.py       # Dataset loading utilities
├── utils.py                 # Helper functions and utilities
├── example.py               # Usage examples
├── configs/                 # Configuration files
│   └── routerdc_mot_config.json
├── experiments/             # Experiment results and checkpoints
├── logs/                    # Training logs
├── pareval/                 # ParEval benchmark integration
│   ├── generate_pareval.py  # Code generation script
│   ├── evaluate_pareval.py  # Evaluation script
│   ├── drivers/             # ParEval test drivers
│   └── README.md            # Detailed ParEval documentation
└── requirements.txt         # Package dependencies
```

## Usage

### Quick Start

Run a simple inference example:
```bash
python example.py --demo inference
```

### Training

#### Single GPU Training
```bash
python train_ddp.py --config configs/routerdc_mot_config.json
```

#### Multi-GPU Training
```bash
./run_experiments.sh --gpus 0,1,2,3 --exp-name mot_experiment
```

Or using torchrun directly:
```bash
torchrun --nproc_per_node=4 train_ddp.py --config configs/routerdc_mot_config.json
```

### Configuration

The framework is configured through JSON files. Key configuration parameters include:

```json
{
  "experiment_name": "mot_experiment",
  "expert_models": [
    "model_name_1",
    "model_name_2"
  ],
  "mot_config": {
    "num_stacks": 4,
    "top_k": 3,
    "shared_dim": 768,
    "router_hidden_dim": 256,
    "interaction_heads": 8
  },
  "training": {
    "batch_size": 8,
    "learning_rate": 1e-4,
    "num_epochs": 10,
    "gradient_accumulation_steps": 4
  }
}
```

### Example Code

Basic usage example:

```python
from mixture_of_thoughts import MixtureOfThoughts, MoTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load expert models
expert_models = [
    AutoModelForCausalLM.from_pretrained("gpt2"),
    AutoModelForCausalLM.from_pretrained("distilgpt2")
]
tokenizers = [
    AutoTokenizer.from_pretrained("gpt2"),
    AutoTokenizer.from_pretrained("distilgpt2")
]

# Configure MoT
config = MoTConfig(
    num_stacks=4,
    top_k=2,
    shared_dim=768
)

# Initialize framework
mot_model = MixtureOfThoughts(
    expert_models=expert_models,
    tokenizers=tokenizers,
    config=config
)

# Run inference
input_ids = tokenizers[0]("Hello, world!", return_tensors="pt").input_ids
outputs = mot_model(input_ids=input_ids)
```

## Supported Datasets

The framework includes loaders for the following benchmark datasets:

- **MMLU**: Massive Multitask Language Understanding
- **GSM8K**: Grade School Math 8K
- **CMMLU**: Chinese Massive Multitask Language Understanding
- **ARC-Challenge**: AI2 Reasoning Challenge
- **HumanEval**: Code generation benchmark
- **ParEval**: Parallel code generation benchmark (see below)

## ParEval Benchmark Evaluation

The MoT framework includes integration with [ParEval](https://github.com/parallelcodefoundry/ParEval), a benchmark for evaluating parallel code generation capabilities across multiple parallelism models (CUDA, OpenMP, MPI, etc.).

### Quick Start

```bash
cd mot/pareval

# 1. Generate code samples (C++ to CUDA translation)
python generate_pareval.py \
    --prompts ~/ParEval/prompts/translation-prompts.json \
    --task translation \
    --output ./outputs/mot_translation.json \
    --num_samples 50

# 2. Evaluate generated code (compile, run, compute pass@k)
python evaluate_pareval.py --input ./outputs/mot_translation.json
```

### Features

- **Code Generation**: Generate parallel code samples using trained MoT models
- **Automated Evaluation**: Compile and test generated code against ParEval test cases
- **Metrics Computation**: Calculate pass@k, build@k, and other standard metrics
- **Multiple Tasks**: Supports both translation (C++ → CUDA) and direct generation tasks

For detailed usage, configuration options, and troubleshooting, see [`pareval/README.md`](pareval/README.md).

## Training Scripts

### Distributed Training
```bash
./run_experiments.sh [OPTIONS]

Options:
  -g, --gpus          GPU IDs to use (e.g., '0,1,2,3')
  -n, --num-gpus      Number of GPUs to use
  -c, --config        Path to configuration file
  -e, --exp-name      Experiment name for logging
  -w, --wandb-mode    WandB mode: online, offline, disabled
  -r, --resume        Path to checkpoint to resume from
```

### Model Downloading
```bash
python download_experts.py --config configs/routerdc_mot_config.json --max-workers 4
```

## Architecture

The MoT framework consists of several key components:

1. **Router Network**: A learnable MLP that assigns scores to each expert based on input embeddings
2. **Expert Models**: Pre-trained language models that serve as specialized experts
3. **Stack Partitioning**: Divides each expert into Q stacks of layers
4. **Interaction Layers**: Cross-attention mechanisms between selected experts
5. **Output Aggregation**: Combines expert outputs using learned weights

### Loss Functions

The framework uses multiple loss components:
- Primary task loss (e.g., language modeling)
- Router entropy regularization
- Load balancing loss for expert utilization
- Auxiliary expert-specific losses

## Performance Considerations

- Models are cached in `~/.cache/huggingface/hub` by default
- Supports 8-bit quantization for memory efficiency
- Implements gradient checkpointing for large models
- Uses mixed precision training (fp16/bf16) when available

## Logging and Monitoring

- Training logs are saved to `logs/` directory
- Experiment results and checkpoints stored in `experiments/`
- Supports Weights & Biases (wandb) integration for experiment tracking
- Real-time training metrics displayed during training

## Development

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black .
isort .
flake8 .
```

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or enable gradient checkpointing
2. **Model Download Failures**: Check network connection and HuggingFace hub access
3. **DDP Training Issues**: Ensure all GPUs are visible and NCCL is properly installed

### Debug Mode

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```


