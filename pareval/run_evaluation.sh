#!/bin/bash
# =============================================================================
# Run MoT Evaluation on ParEval Benchmark
# =============================================================================
# This script runs the complete evaluation pipeline:
# 1. Generate code samples using MoT
# 2. Run ParEval evaluation (compile and run generated code)
# 3. Compute metrics
#
# Usage:
#   ./run_evaluation.sh [OPTIONS]
#
# Examples:
#   # Run translation task evaluation
#   ./run_evaluation.sh --task translation
#
#   # Run generation task evaluation with custom settings
#   ./run_evaluation.sh --task generation --num_samples 10 --temperature 0.5
#
#   # Run with full MoT generation mode
#   ./run_evaluation.sh --task translation --use_mot_generate
# =============================================================================

set -e  # Exit on error

# =============================================================================
# Default Configuration
# =============================================================================
TASK="translation"
PARALLELISM_MODEL="cuda"
NUM_SAMPLES=50
TEMPERATURE=0.2
TOP_P=0.95
MAX_NEW_TOKENS=1024
USE_MOT_GENERATE=""
CHECKPOINT="../cuda_mot_output/best_model.pt"
SEED=42
SKIP_GENERATE=false
SKIP_EVAL=false
DRY_RUN=false

# Paths (relative to this script's directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOT_DIR="$(dirname "$SCRIPT_DIR")"
PAREVAL_DIR="$HOME/ParEval"
OUTPUT_DIR="$SCRIPT_DIR/outputs"

# =============================================================================
# Parse Arguments
# =============================================================================
print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --task TASK              Task type: 'generation' or 'translation' (default: $TASK)"
    echo "  --parallelism_model PM   Parallelism model filter (default: $PARALLELISM_MODEL)"
    echo "  --num_samples N          Number of samples per prompt (default: $NUM_SAMPLES)"
    echo "  --temperature T          Sampling temperature (default: $TEMPERATURE)"
    echo "  --top_p P                Top-p sampling parameter (default: $TOP_P)"
    echo "  --max_new_tokens N       Max new tokens to generate (default: $MAX_NEW_TOKENS)"
    echo "  --use_mot_generate       Use full MoT generation mode (slower)"
    echo "  --checkpoint PATH        Path to MoT checkpoint (default: $CHECKPOINT)"
    echo "  --seed N                 Random seed (default: $SEED)"
    echo "  --pareval_dir PATH       Path to ParEval directory (default: $PAREVAL_DIR)"
    echo "  --output_dir PATH        Output directory (default: $OUTPUT_DIR)"
    echo "  --skip_generate          Skip code generation step"
    echo "  --skip_eval              Skip ParEval evaluation step"
    echo "  --dry_run                Print commands without executing"
    echo "  --help                   Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --task translation --num_samples 50"
    echo "  $0 --task generation --use_mot_generate --temperature 0.5"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --task)
            TASK="$2"
            shift 2
            ;;
        --parallelism_model)
            PARALLELISM_MODEL="$2"
            shift 2
            ;;
        --num_samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --top_p)
            TOP_P="$2"
            shift 2
            ;;
        --max_new_tokens)
            MAX_NEW_TOKENS="$2"
            shift 2
            ;;
        --use_mot_generate)
            USE_MOT_GENERATE="--use_mot_generate"
            shift
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --pareval_dir)
            PAREVAL_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --skip_generate)
            SKIP_GENERATE=true
            shift
            ;;
        --skip_eval)
            SKIP_EVAL=true
            shift
            ;;
        --dry_run)
            DRY_RUN=true
            shift
            ;;
        --help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# =============================================================================
# Validate Configuration
# =============================================================================
echo "=============================================================="
echo "MoT Evaluation on ParEval"
echo "=============================================================="
echo "Configuration:"
echo "  Task:              $TASK"
echo "  Parallelism model: $PARALLELISM_MODEL"
echo "  Num samples:       $NUM_SAMPLES"
echo "  Temperature:       $TEMPERATURE"
echo "  Top-p:             $TOP_P"
echo "  Max new tokens:    $MAX_NEW_TOKENS"
echo "  MoT generate mode: $([ -n "$USE_MOT_GENERATE" ] && echo 'Yes' || echo 'No')"
echo "  Checkpoint:        $CHECKPOINT"
echo "  Seed:              $SEED"
echo "  ParEval dir:       $PAREVAL_DIR"
echo "  Output dir:        $OUTPUT_DIR"
echo "=============================================================="

# Validate task
if [[ "$TASK" != "generation" && "$TASK" != "translation" ]]; then
    echo "Error: Invalid task '$TASK'. Must be 'generation' or 'translation'."
    exit 1
fi

# Set prompts file based on task
if [[ "$TASK" == "translation" ]]; then
    PROMPTS_FILE="$PAREVAL_DIR/prompts/translation-prompts.json"
else
    PROMPTS_FILE="$PAREVAL_DIR/prompts/generation-prompts.json"
fi

# Validate paths
if [[ ! -f "$PROMPTS_FILE" ]]; then
    echo "Error: Prompts file not found: $PROMPTS_FILE"
    exit 1
fi

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Error: Checkpoint file not found: $CHECKPOINT"
    exit 1
fi

if [[ ! -d "$PAREVAL_DIR/drivers" ]]; then
    echo "Error: ParEval drivers directory not found: $PAREVAL_DIR/drivers"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Define output file names
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_PREFIX="${TASK}_${PARALLELISM_MODEL}_n${NUM_SAMPLES}_t${TEMPERATURE}"
GENERATED_FILE="$OUTPUT_DIR/${OUTPUT_PREFIX}_generated.json"
RESULTS_FILE="$OUTPUT_DIR/${OUTPUT_PREFIX}_results.json"

echo ""
echo "Output files:"
echo "  Generated code: $GENERATED_FILE"
echo "  Eval results:   $RESULTS_FILE"
echo ""

# =============================================================================
# Step 1: Generate Code Samples
# =============================================================================
if [[ "$SKIP_GENERATE" == false ]]; then
    echo "=============================================================="
    echo "Step 1: Generating Code Samples"
    echo "=============================================================="
    
    GENERATE_CMD="python $SCRIPT_DIR/generate_pareval.py \
        --prompts $PROMPTS_FILE \
        --task $TASK \
        --output $GENERATED_FILE \
        --checkpoint $CHECKPOINT \
        --parallelism_model $PARALLELISM_MODEL \
        --num_samples $NUM_SAMPLES \
        --temperature $TEMPERATURE \
        --top_p $TOP_P \
        --max_new_tokens $MAX_NEW_TOKENS \
        --seed $SEED \
        $USE_MOT_GENERATE"
    
    echo "Running: $GENERATE_CMD"
    echo ""
    
    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY RUN] Would execute the above command"
    else
        eval $GENERATE_CMD
    fi
    
    echo ""
else
    echo "Skipping code generation step (--skip_generate)"
    if [[ ! -f "$GENERATED_FILE" ]]; then
        echo "Error: Generated file not found: $GENERATED_FILE"
        echo "Cannot skip generation step without existing generated file."
        exit 1
    fi
fi

# =============================================================================
# Step 2: Run ParEval Evaluation
# =============================================================================
if [[ "$SKIP_EVAL" == false ]]; then
    echo "=============================================================="
    echo "Step 2: Running ParEval Evaluation"
    echo "=============================================================="
    
    EVAL_CMD="cd $PAREVAL_DIR/drivers && python run-all.py $GENERATED_FILE -o $RESULTS_FILE --yes-to-all"
    
    echo "Running: $EVAL_CMD"
    echo ""
    echo "WARNING: This will compile and run generated code."
    echo "Make sure you are in a safe environment."
    echo ""
    
    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY RUN] Would execute the above command"
    else
        eval $EVAL_CMD
    fi
    
    echo ""
else
    echo "Skipping ParEval evaluation step (--skip_eval)"
fi

# =============================================================================
# Summary
# =============================================================================
echo "=============================================================="
echo "Evaluation Complete"
echo "=============================================================="
echo ""
echo "Generated files:"
echo "  - $GENERATED_FILE"
if [[ -f "$RESULTS_FILE" ]]; then
    echo "  - $RESULTS_FILE"
fi
echo ""
echo "To compute metrics, run:"
echo "  cd $PAREVAL_DIR/analysis"
echo "  python metrics.py <results_csv> --model-name MoT-${TASK}"
echo ""
echo "=============================================================="