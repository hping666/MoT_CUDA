# CUDA-MoT: Mixture of Thoughts for C++ to CUDA Code Translation

A framework that applies **Mixture of Thoughts (MoT)** to translate C++ code into CUDA code by orchestrating multiple expert code generation models with learned sparse routing.

## Overview

CUDA-MoT combines multiple pre-trained code LLMs (experts) through a trainable router that learns to select the best expert(s) for each input. Only the router and interaction layers are trained—expert models remain frozen, enabling efficient fine-tuning.

### Expert Models (Default)
- **Qwen2.5-Coder** – General code generation
- **HPC-Coder-v2** – Specialized for HPC/parallel code  
- **StarCoder2** – Multi-language code generation

## Architecture

```
Input (C++ Code)
      ↓
┌─────────────────┐
│ Sentence Encoder│  (DeBERTa-v3)
└────────┬────────┘
         ↓
┌─────────────────┐
│  Sparse Router  │  → Select top-k experts
└────────┬────────┘
         ↓
┌─────────────────┐
│  Expert Models  │  (Frozen LLMs)
│  Stack-based    │
│  Processing     │
└────────┬────────┘
         ↓
┌─────────────────┐
│Interaction Layer│  → Combine expert outputs
└────────┬────────┘
         ↓
Output (CUDA Code)
```

## Installation

```bash
pip install -r requirements.txt
```

**Key dependencies:** PyTorch, Transformers, sentence-transformers, bitsandbytes (for quantization)

## Usage

### Training

```bash
python train_cuda_mot.py \
    --data_path ./cpp_cuda_train.jsonl \
    --num_epochs 3 \
    --batch_size 1 \
    --use_4bit \
    --output_dir ./cuda_mot_output
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_path` | `./cpp_cuda_train_synthetic.jsonl` | Path to training data (JSONL) |
| `--use_4bit` | `True` | Enable 4-bit quantization |
| `--use_8bit` | `False` | Enable 8-bit quantization |
| `--num_stacks` | `4` | Number of layer stacks per expert |
| `--top_k` | `2` | Number of experts to activate |
| `--learning_rate` | `5e-6` | Learning rate |
| `--max_length` | `1024` | Maximum sequence length |
| `--eval_only` | `False` | Run evaluation only |

### Evaluation Only

```bash
python train_cuda_mot.py \
    --eval_only \
    --checkpoint_path ./cuda_mot_output/best_model.pt
```

## Data Format

JSONL file with each line containing:
```json
{"cpp": "void add(int* a, int* b, int n) {...}", "generated_cuda": "__global__ void add_kernel(...) {...}"}
```

## Evaluation Metrics

- **BLEU-4** – N-gram precision
- **chrF** – Character-level F-score  
- **ROUGE-L** – Longest common subsequence
- **Exact Match** – Normalized exact match rate
- **Edit Similarity** – 1 - normalized edit distance
- **Syntax Validity** – Basic CUDA syntax check

## Project Structure

```
├── train_cuda_mot.py      # Main training script
├── mixture_of_thoughts.py # MoT framework implementation
├── cuda_dataset.py        # Dataset loading and processing
├── cuda_evaluation.py     # Evaluation metrics
├── utils.py               # Utilities (model loading, checkpoints)
└── requirements.txt       # Dependencies
```

## License

MIT License
