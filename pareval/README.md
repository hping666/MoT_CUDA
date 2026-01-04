# MoT ParEval Integration

This module provides tools to evaluate Mixture of Thoughts (MoT) models on the [ParEval](https://github.com/parallelcodefoundry/ParEval) benchmark for parallel code generation.

## Overview

The ParEval benchmark evaluates code generation models on their ability to generate correct and efficient parallel code across multiple parallelism models (CUDA, OpenMP, MPI, etc.). This integration allows you to:

1. Generate code samples using a trained MoT model
2. Evaluate the generated code using ParEval's test infrastructure
3. Compute standard metrics (pass@k, build@k, speedup@k, etc.)

## Directory Structure

```
mot/pareval/
├── __init__.py              # Package initialization
├── generate_pareval.py      # Main generation script
├── evaluate_pareval.py      # Evaluation script (compile, run, compute metrics)
├── run_evaluation.sh        # Complete evaluation pipeline script
├── drivers/                 # ParEval drivers (copied from ~/ParEval/drivers)
│   ├── cpp/
│   ├── util.py
│   ├── build-configs.json
│   ├── launch-configs.json
│   ├── problem-sizes.json
│   └── ...
└── README.md                # This documentation
```

## Requirements

- Trained MoT checkpoint (default: `../cuda_mot_output/best_model.pt`)
- ParEval drivers copied to `./drivers/` (from `~/ParEval/drivers`)
- CUDA compiler (nvcc) for running CUDA evaluations
- Python dependencies from MoT project

## Quick Start

### 1. Generate Code Samples

```bash
cd ~/mot/pareval

# For translation task (C++ to CUDA translation)
python generate_pareval.py \
    --prompts ~/ParEval/prompts/translation-prompts.json \
    --task translation \
    --output ./outputs/mot_translation.json \
    --parallelism_model cuda \
    --num_samples 50

# For generation task (direct CUDA generation)
python generate_pareval.py \
    --prompts ~/ParEval/prompts/generation-prompts.json \
    --task generation \
    --output ./outputs/mot_generation.json \
    --parallelism_model cuda \
    --num_samples 50
```

### 2. Run Evaluation (Compile and Test)

```bash
cd ~/mot/pareval

# Evaluate generated code (compile, run, compute metrics)
python evaluate_pareval.py --input ./outputs/mot_translation.json

# With custom k values for pass@k computation
python evaluate_pareval.py --input ./outputs/mot_translation.json \
    --k_values 1 5 10 50

# Only compute metrics from existing results (skip compile/run)
python evaluate_pareval.py --input ./outputs/mot_translation_results.json \
    --metrics_only
```

### 3. (Alternative) Run ParEval Evaluation Directly

```bash
cd ~/ParEval/drivers

# Evaluate generated code
python run-all.py ~/mot/pareval/outputs/mot_translation.json \
    -o ~/mot/pareval/outputs/mot_translation_results.json \
    --yes-to-all
```

### 3. Compute Metrics

```bash
cd ~/ParEval/analysis

# Convert results to CSV and compute metrics
python metrics.py <results_csv_file> --model-name MoT-CUDA
```

## Using the Shell Script

For convenience, you can use the provided shell script to run the complete pipeline:

```bash
cd ~/mot/pareval

# Run translation task evaluation
./run_evaluation.sh --task translation

# Run generation task evaluation
./run_evaluation.sh --task generation

# Run with custom settings
./run_evaluation.sh \
    --task translation \
    --num_samples 10 \
    --temperature 0.5 \
    --use_mot_generate
```

## Command Line Arguments

### evaluate_pareval.py

| Argument | Description | Default |
|----------|-------------|---------|
| `--input` | Path to generated code JSON file | Required |
| `--output` | Path to output results JSON file | `<input>_results.json` |
| `--metrics_output` | Path to output metrics JSON file | `<input>_metrics.json` |
| `--drivers_dir` | Path to local drivers directory | `./drivers` |
| `--metrics_only` | Skip compilation, only compute metrics | `False` |
| `--skip_metrics` | Skip metrics computation | `False` |
| `--local` | Use local execution (no SLURM/srun) | `True` |
| `--use_slurm` | Use SLURM launch configs (requires srun) | `False` |
| `--k_values` | K values for pass@k computation | `1 5 10` |
| `--model_name` | Model name for metrics output | `MoT-CUDA` |
| `--build_timeout` | Build timeout in seconds | `60` |
| `--run_timeout` | Run timeout in seconds | `120` |
| `--log_level` | Logging level | `INFO` |
| `--log_build_errors` | Display detailed build errors | `False` |

### generate_pareval.py

| Argument | Description | Default |
|----------|-------------|---------|
| `--prompts` | Path to ParEval prompts JSON file | Required |
| `--task` | Task type: `generation` or `translation` | Required |
| `--output` | Path to output JSON file | Required |
| `--checkpoint` | Path to MoT checkpoint | `../cuda_mot_output/best_model.pt` |
| `--parallelism_model` | Filter prompts by parallelism model | `cuda` |
| `--num_samples` | Number of samples per prompt | `50` |
| `--temperature` | Sampling temperature | `0.2` |
| `--top_p` | Top-p sampling parameter | `0.95` |
| `--max_new_tokens` | Maximum new tokens to generate | `1024` |
| `--use_mot_generate` | Use full MoT generation mode | `False` |
| `--use_4bit` | Use 4-bit quantization | `True` |
| `--use_8bit` | Use 8-bit quantization | `False` |
| `--no_quantization` | Disable quantization | `False` |
| `--seed` | Random seed | `42` |

## Generation Modes

### Fast Mode (default)
Uses the MoT router to select the primary expert, then generates using that expert's native generation. Faster but doesn't utilize expert interaction during generation.

```bash
python generate_pareval.py --task translation ...
```

### Full MoT Mode
Each token is generated using the full MoT framework with expert interaction. Slower but more accurately reflects the MoT architecture.

```bash
python generate_pareval.py --task translation --use_mot_generate ...
```

## Configuring Expert Models

To change the expert models used by MoT, edit the `EXPERT_MODELS` list in `generate_pareval.py`:

```python
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
```

**Important**: The expert models must match the ones used during MoT training.

## Output Format

The generated output JSON is compatible with ParEval's `run-all.py`:

```json
[
    {
        "name": "problem_name",
        "parallelism_model": "cuda",
        "problem_type": "geometry",
        "language": "cpp",
        "prompt": "...",
        "outputs": ["generated_code_1", "generated_code_2", ...],
        "temperature": 0.2,
        "top_p": 0.95,
        ...
    },
    ...
]
```

## Troubleshooting

### CUDA Out of Memory
- Try using `--use_4bit` quantization (default)
- Reduce `--max_new_tokens`
- Use smaller expert models (edit `EXPERT_MODELS` in `generate_pareval.py`)

### Checkpoint Not Found
- Ensure the MoT checkpoint exists at the specified path
- Default path is `../cuda_mot_output/best_model.pt`

### ParEval Evaluation Fails
- Ensure CUDA compiler (nvcc) is available
- Check that ParEval drivers are properly copied to `./drivers/`
- Verify the generated code format matches ParEval expectations

### Import Errors
- Run the script from `~/mot/pareval` directory
- Ensure all MoT dependencies are installed

## Notes

- The MoT model was trained on CUDA code translation, so best results are expected with `--parallelism_model cuda`
- ParEval's default sampling parameters (temperature=0.2, num_samples=50) are used by default for fair comparison
- The `--use_mot_generate` mode is slower but provides a more complete evaluation of the MoT architecture

## References

- [ParEval Repository](https://github.com/parallelcodefoundry/ParEval)
- [MoT Framework](../README.md)
