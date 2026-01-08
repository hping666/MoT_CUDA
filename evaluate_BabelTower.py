"""
Evaluation script for MoT framework on BabelTower dataset.
Evaluates C-to-CUDA translation using metrics from BabelTower paper:
- BLEU, CodeBLEU, ParaBLEU, and Compilation Accuracy.
"""

import os
import sys
import json
import argparse
import re
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter
from dataclasses import dataclass

import torch
from tqdm import tqdm
from transformers import set_seed

from mixture_of_thoughts import MixtureOfThoughts, MoTConfig
from utils import ExpertConfig, ExpertLoader


# =============================================================================
# Default Configuration
# =============================================================================
DEFAULT_CONFIG = {
    'dataset_dir': './BabelTower/dataset',
    'checkpoint_path': './cuda_mot_output/best_model.pt',
    'output_dir': './babeltower_eval_output',
    'use_8bit': False,
    'use_4bit': True,
    'single_gpu': True,
    'max_new_tokens': 256,
    'temperature': 0.7,
    'top_p': 0.95,
    'use_mot_generate': False,
    'num_samples': None, # None = all samples
    'seed': 42,
    'check_compilation': True,
    'save_predictions': True,
}

EXPERT_MODELS = [
    {'name': 'Qwen/Qwen2.5-Coder-1.5B', 'description': 'Qwen2.5 Coder 1.5B'},
    {'name': 'hpcgroup/hpc-coder-v2-1.3b', 'description': 'HPC-Coder-v2 1.3B'},
    {'name': 'bigcode/starcoder2-3b', 'description': 'StarCoder2 3B'},
]


# =============================================================================
# BabelTower Dataset Loader
# =============================================================================
@dataclass
class BabelTowerSample:
    """Single sample from BabelTower dataset."""
    cpp_code: str
    cuda_code: str
    index: int


def load_babeltower_dataset(dataset_dir: str, split: str = 'test') -> List[BabelTowerSample]:
    """
    Load BabelTower dataset from .tok files.
    
    Args:
        dataset_dir: Path to dataset directory
        split: 'test' or 'valid'
    
    Returns:
        List of BabelTowerSample objects
    """
    dataset_path = Path(dataset_dir)
    cpp_file = dataset_path / f'cpp.para.{split}.tok'
    cuda_file = dataset_path / f'cuda.para.{split}.tok'
    
    if not cpp_file.exists():
        raise FileNotFoundError(f"C++ file not found: {cpp_file}")
    if not cuda_file.exists():
        raise FileNotFoundError(f"CUDA file not found: {cuda_file}")
    
    with open(cpp_file, 'r', encoding='utf-8') as f:
        cpp_lines = [line.strip() for line in f.readlines()]
    
    with open(cuda_file, 'r', encoding='utf-8') as f:
        cuda_lines = [line.strip() for line in f.readlines()]
    
    if len(cpp_lines) != len(cuda_lines):
        raise ValueError(f"Mismatch: {len(cpp_lines)} C++ vs {len(cuda_lines)} CUDA samples")
    
    samples = []
    for i, (cpp, cuda) in enumerate(zip(cpp_lines, cuda_lines)):
        # Detokenize: BabelTower uses space-separated tokens
        cpp_code = detokenize_code(cpp)
        cuda_code = detokenize_code(cuda)
        samples.append(BabelTowerSample(cpp_code=cpp_code, cuda_code=cuda_code, index=i))
    
    return samples


def detokenize_code(tokenized: str) -> str:
    """
    Convert space-separated tokens back to code.
    Handles common code patterns.
    """
    code = tokenized
    
    # Fix spacing around operators and punctuation
    code = re.sub(r'\s*\.\s*', '.', code)
    code = re.sub(r'\s*->\s*', '->', code)
    code = re.sub(r'\s*::\s*', '::', code)
    code = re.sub(r'\s*;\s*', '; ', code)
    code = re.sub(r'\s*,\s*', ', ', code)
    code = re.sub(r'\s*\(\s*', '(', code)
    code = re.sub(r'\s*\)\s*', ') ', code)
    code = re.sub(r'\s*\[\s*', '[', code)
    code = re.sub(r'\s*\]\s*', '] ', code)
    code = re.sub(r'\s*{\s*', ' { ', code)
    code = re.sub(r'\s*}\s*', ' } ', code)
    code = re.sub(r'\s*<\s*', ' < ', code)
    code = re.sub(r'\s*>\s*', ' > ', code)
    code = re.sub(r'\s*\+\s*\+\s*', '++ ', code)
    code = re.sub(r'\s*-\s*-\s*', '-- ', code)
    code = re.sub(r'\s*\+\s*=\s*', ' += ', code)
    code = re.sub(r'\s*-\s*=\s*', ' -= ', code)
    code = re.sub(r'\s*\*\s*=\s*', ' *= ', code)
    code = re.sub(r'\s*&\s*&\s*', ' && ', code)
    code = re.sub(r'\s*\|\s*\|\s*', ' || ', code)
    
    # Clean up multiple spaces
    code = re.sub(r'\s+', ' ', code)
    code = code.strip()
    
    return code


# =============================================================================
# BabelTower Metrics Implementation
# =============================================================================
class BabelTowerEvaluator:
    """
    Evaluator implementing metrics from BabelTower paper:
    BLEU, CodeBLEU, ParaBLEU, and Compilation Accuracy.
    """
    
    # CUDA-specific keywords for ParaBLEU
    CUDA_KEYWORDS = [
        '__global__', '__device__', '__shared__', '__constant__',
        'threadIdx', 'blockIdx', 'blockDim', 'gridDim',
        'atomicAdd', 'atomicSub', 'atomicExch', 'atomicMin', 'atomicMax',
        '__syncthreads', 'cudaMalloc', 'cudaFree', 'cudaMemcpy'
    ]
    
    def __init__(self, check_compilation: bool = True):
        self.check_compilation = check_compilation
    
    def compute_bleu(
        self,
        predictions: List[str],
        references: List[str],
        max_n: int = 4
    ) -> float:
        """Compute BLEU score (0-100 scale)."""
        if not predictions or len(predictions) != len(references):
            return 0.0
        
        weights = [1.0 / max_n] * max_n
        clipped_counts = [0] * max_n
        total_counts = [0] * max_n
        ref_lengths = 0
        pred_lengths = 0
        
        for pred, ref in zip(predictions, references):
            pred_tokens = self._tokenize(pred)
            ref_tokens = self._tokenize(ref)
            
            pred_lengths += len(pred_tokens)
            ref_lengths += len(ref_tokens)
            
            for n in range(1, max_n + 1):
                pred_ngrams = self._get_ngrams(pred_tokens, n)
                ref_ngrams = self._get_ngrams(ref_tokens, n)
                
                for ngram, count in pred_ngrams.items():
                    clipped_counts[n-1] += min(count, ref_ngrams.get(ngram, 0))
                    total_counts[n-1] += count
        
        precisions = []
        for n in range(max_n):
            if total_counts[n] > 0:
                precisions.append(clipped_counts[n] / total_counts[n])
            else:
                precisions.append(0.0)
        
        if min(precisions) > 0:
            log_precision = sum(w * math.log(p) for w, p in zip(weights, precisions))
            geo_mean = math.exp(log_precision)
        else:
            geo_mean = 0.0
        
        if pred_lengths > 0:
            bp = math.exp(min(0, 1 - ref_lengths / pred_lengths))
        else:
            bp = 0.0
        
        return bp * geo_mean * 100
    
    def compute_codebleu(
        self,
        predictions: List[str],
        references: List[str],
        alpha: float = 0.25,
        beta: float = 0.25,
        gamma: float = 0.25,
        delta: float = 0.25
    ) -> float:
        """
        Compute CodeBLEU score.
        CodeBLEU = alpha * BLEU + beta * BLEU_weight + gamma * Match_ast + delta * Match_df
        
        Simplified version: uses weighted n-gram and keyword matching.
        """
        if not predictions:
            return 0.0
        
        # Standard BLEU
        bleu = self.compute_bleu(predictions, references)
        
        # Weighted BLEU (emphasize code keywords)
        bleu_weight = self._compute_weighted_bleu(predictions, references)
        
        # AST match approximation (structural similarity)
        ast_match = self._compute_ast_match(predictions, references)
        
        # Data flow match approximation
        df_match = self._compute_dataflow_match(predictions, references)
        
        codebleu = alpha * bleu + beta * bleu_weight + gamma * ast_match + delta * df_match
        return codebleu
    
    def compute_parableu(
        self,
        predictions: List[str],
        references: List[str],
        alpha: float = 0.25,
        beta: float = 0.25,
        gamma: float = 0.25,
        delta: float = 0.25
    ) -> float:
        """
        Compute ParaBLEU score (BabelTower's parallel semantics metric).
        ParaBLEU = (alpha*BLEU + beta*BLEU_weight + gamma*Match_ast + delta*Match_df)
                   * SIM_cuda_keywords * SIM_loops * SIM_parallel
        """
        if not predictions:
            return 0.0
        
        # Base CodeBLEU components
        bleu = self.compute_bleu(predictions, references)
        bleu_weight = self._compute_weighted_bleu(predictions, references)
        ast_match = self._compute_ast_match(predictions, references)
        df_match = self._compute_dataflow_match(predictions, references)
        
        base_score = alpha * bleu + beta * bleu_weight + gamma * ast_match + delta * df_match
        
        # Parallel semantics penalties
        sim_cuda = self._compute_cuda_keyword_similarity(predictions, references)
        sim_loops = self._compute_loop_similarity(predictions, references)
        sim_parallel = self._compute_parallel_similarity(predictions, references)
        
        parableu = base_score * sim_cuda * sim_loops * sim_parallel
        return parableu
    
    def compute_compilation_accuracy(self, predictions: List[str]) -> float:
        """Check compilation accuracy of generated CUDA code."""
        if not self.check_compilation or not predictions:
            return 0.0
        
        compiled = 0
        for pred in predictions:
            if self._check_cuda_syntax(pred):
                compiled += 1
        
        return (compiled / len(predictions)) * 100
    
    def evaluate_all(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """Compute all BabelTower metrics."""
        results = {
            'bleu': self.compute_bleu(predictions, references),
            'codebleu': self.compute_codebleu(predictions, references),
            'parableu': self.compute_parableu(predictions, references),
            'compilation_accuracy': self.compute_compilation_accuracy(predictions),
            'num_samples': len(predictions),
        }
        return results
    
    def print_results(self, results: Dict[str, float]) -> None:
        """Print evaluation results."""
        print("\n" + "=" * 60)
        print("BabelTower Evaluation Results")
        print("=" * 60)
        print(f"  Samples evaluated: {results.get('num_samples', 'N/A')}")
        print("-" * 60)
        print(f"  BLEU:                  {results.get('bleu', 0):.2f}")
        print(f"  CodeBLEU:              {results.get('codebleu', 0):.2f}")
        print(f"  ParaBLEU:              {results.get('parableu', 0):.2f}")
        print(f"  Compilation Accuracy:  {results.get('compilation_accuracy', 0):.2f}%")
        print("=" * 60 + "\n")
    
    # ----- Helper Methods -----
    
    def _tokenize(self, code: str) -> List[str]:
        """Tokenize code for BLEU computation."""
        code = re.sub(r'[ \t]+', ' ', code)
        tokens = re.findall(r'\w+|[^\w\s]', code)
        return tokens
    
    def _get_ngrams(self, tokens: List[str], n: int) -> Counter:
        """Get n-gram counts."""
        ngrams = Counter()
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngrams[ngram] += 1
        return ngrams
    
    def _compute_weighted_bleu(
        self,
        predictions: List[str],
        references: List[str]
    ) -> float:
        """Compute weighted BLEU emphasizing code keywords."""
        keywords = set([
            'int', 'float', 'double', 'void', 'char', 'long', 'short',
            'for', 'while', 'if', 'else', 'return', 'const', 'static',
            '__global__', '__device__', '__shared__', 'threadIdx', 'blockIdx'
        ])
        
        total_match = 0
        total_ref = 0
        
        for pred, ref in zip(predictions, references):
            pred_tokens = set(self._tokenize(pred))
            ref_tokens = set(self._tokenize(ref))
            
            pred_kw = pred_tokens & keywords
            ref_kw = ref_tokens & keywords
            
            if ref_kw:
                total_match += len(pred_kw & ref_kw)
                total_ref += len(ref_kw)
        
        if total_ref == 0:
            return 0.0
        return (total_match / total_ref) * 100
    
    def _compute_ast_match(
        self,
        predictions: List[str],
        references: List[str]
    ) -> float:
        """Approximate AST match using structural patterns."""
        patterns = [
            r'for\s*\([^)]+\)',      # for loops
            r'while\s*\([^)]+\)',    # while loops
            r'if\s*\([^)]+\)',       # if statements
            r'\w+\s*\([^)]*\)\s*{',  # function definitions
            r'\w+\s*\[[^\]]+\]',     # array access
        ]
        
        total_match = 0
        total_ref = 0
        
        for pred, ref in zip(predictions, references):
            for pattern in patterns:
                pred_matches = set(re.findall(pattern, pred))
                ref_matches = set(re.findall(pattern, ref))
                
                if ref_matches:
                    total_match += len(pred_matches & ref_matches)
                    total_ref += len(ref_matches)
        
        if total_ref == 0:
            return 50.0  # Neutral score
        return (total_match / total_ref) * 100
    
    def _compute_dataflow_match(
        self,
        predictions: List[str],
        references: List[str]
    ) -> float:
        """Approximate data flow match using variable patterns."""
        total_sim = 0.0
        
        for pred, ref in zip(predictions, references):
            # Extract variable-like identifiers
            pred_vars = set(re.findall(r'\b[a-zA-Z_]\w*\b', pred))
            ref_vars = set(re.findall(r'\b[a-zA-Z_]\w*\b', ref))
            
            # Remove common keywords
            common_kw = {'int', 'float', 'void', 'for', 'if', 'else', 'return'}
            pred_vars -= common_kw
            ref_vars -= common_kw
            
            if ref_vars:
                overlap = len(pred_vars & ref_vars)
                total_sim += overlap / len(ref_vars)
        
        if not predictions:
            return 0.0
        return (total_sim / len(predictions)) * 100
    
    def _compute_cuda_keyword_similarity(
        self,
        predictions: List[str],
        references: List[str]
    ) -> float:
        """Compute CUDA keyword similarity factor."""
        total_sim = 0.0
        
        for pred, ref in zip(predictions, references):
            pred_kw = set(kw for kw in self.CUDA_KEYWORDS if kw in pred)
            ref_kw = set(kw for kw in self.CUDA_KEYWORDS if kw in ref)
            
            if ref_kw:
                sim = len(pred_kw & ref_kw) / len(ref_kw)
            else:
                sim = 1.0 if not pred_kw else 0.5
            
            total_sim += sim
        
        if not predictions:
            return 1.0
        return total_sim / len(predictions)
    
    def _compute_loop_similarity(
        self,
        predictions: List[str],
        references: List[str]
    ) -> float:
        """Compute loop structure similarity."""
        total_sim = 0.0
        
        for pred, ref in zip(predictions, references):
            pred_loops = len(re.findall(r'\bfor\s*\(', pred))
            ref_loops = len(re.findall(r'\bfor\s*\(', ref))
            
            if ref_loops == 0 and pred_loops == 0:
                sim = 1.0
            elif ref_loops == 0:
                sim = 0.5
            else:
                sim = 1.0 - abs(pred_loops - ref_loops) / max(pred_loops, ref_loops)
                sim = max(0.0, sim)
            
            total_sim += sim
        
        if not predictions:
            return 1.0
        return total_sim / len(predictions)
    
    def _compute_parallel_similarity(
        self,
        predictions: List[str],
        references: List[str]
    ) -> float:
        """Compute parallel semantics similarity (thread indexing patterns)."""
        thread_patterns = [
            r'threadIdx\s*\.\s*[xyz]',
            r'blockIdx\s*\.\s*[xyz]',
            r'blockDim\s*\.\s*[xyz]',
            r'gridDim\s*\.\s*[xyz]',
        ]
        
        total_sim = 0.0
        
        for pred, ref in zip(predictions, references):
            pred_parallel = []
            ref_parallel = []
            
            for pattern in thread_patterns:
                pred_parallel.extend(re.findall(pattern, pred))
                ref_parallel.extend(re.findall(pattern, ref))
            
            pred_set = set(pred_parallel)
            ref_set = set(ref_parallel)
            
            if not ref_set:
                sim = 1.0 if not pred_set else 0.5
            else:
                # Use Levenshtein-like similarity
                overlap = len(pred_set & ref_set)
                union = len(pred_set | ref_set)
                sim = overlap / union if union > 0 else 0.0
            
            total_sim += sim
        
        if not predictions:
            return 1.0
        return total_sim / len(predictions)
    
    def _check_cuda_syntax(self, code: str) -> bool:
        """Basic CUDA syntax validity check."""
        # Check for CUDA kernel indicator
        has_cuda = '__global__' in code or '__device__' in code
        
        # Check balanced braces
        brace_count = 0
        paren_count = 0
        for char in code:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
            elif char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
            
            if brace_count < 0 or paren_count < 0:
                return False
        
        balanced = (brace_count == 0 and paren_count == 0)
        
        # Check function structure
        has_function = re.search(r'\w+\s*\([^)]*\)\s*{', code) is not None
        
        return has_cuda and balanced and has_function


# =============================================================================
# Model Loading and Evaluation
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate MoT on BabelTower dataset')
    parser.add_argument('--dataset_dir', type=str, default=DEFAULT_CONFIG['dataset_dir'])
    parser.add_argument('--checkpoint_path', type=str, default=DEFAULT_CONFIG['checkpoint_path'])
    parser.add_argument('--output_dir', type=str, default=DEFAULT_CONFIG['output_dir'])
    parser.add_argument('--use_8bit', action='store_true', default=DEFAULT_CONFIG['use_8bit'])
    parser.add_argument('--use_4bit', action='store_true', default=DEFAULT_CONFIG['use_4bit'])
    parser.add_argument('--single_gpu', action='store_true', default=DEFAULT_CONFIG['single_gpu'])
    parser.add_argument('--max_new_tokens', type=int, default=DEFAULT_CONFIG['max_new_tokens'])
    parser.add_argument('--temperature', type=float, default=DEFAULT_CONFIG['temperature'])
    parser.add_argument('--top_p', type=float, default=DEFAULT_CONFIG['top_p'])
    parser.add_argument('--use_mot_generate', action='store_true', default=DEFAULT_CONFIG['use_mot_generate'])
    parser.add_argument('--no_mot_generate', action='store_true')
    parser.add_argument('--num_samples', type=int, default=DEFAULT_CONFIG['num_samples'])
    parser.add_argument('--seed', type=int, default=DEFAULT_CONFIG['seed'])
    parser.add_argument('--check_compilation', action='store_true', default=DEFAULT_CONFIG['check_compilation'])
    parser.add_argument('--save_predictions', action='store_true', default=DEFAULT_CONFIG['save_predictions'])
    parser.add_argument('--split', type=str, default='test', choices=['test', 'valid'])
    return parser.parse_args()


def load_expert_models(args):
    """Load expert models for MoT."""
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


def load_mot_from_checkpoint(checkpoint_path: str, expert_models, tokenizers, device):
    """Load MoT model from checkpoint."""
    print(f"\nLoading checkpoint from {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    mot_config = checkpoint.get('mot_config', {})
    
    # Get hidden dims
    hidden_dims = [m.config.hidden_size for m in expert_models]
    shared_dim = mot_config.get('shared_dim', min(hidden_dims))
    
    config = MoTConfig(
        num_stacks=mot_config.get('num_stacks', 4),
        top_k=mot_config.get('top_k', 2),
        shared_dim=shared_dim,
        router_hidden_dim=mot_config.get('router_hidden_dim', 512),
        router_temperature=mot_config.get('router_temperature', 1.0),
        interaction_heads=mot_config.get('interaction_heads', 8),
        dropout_rate=mot_config.get('dropout_rate', 0.1),
        use_gumbel=mot_config.get('use_gumbel', True),
        gumbel_temperature=mot_config.get('gumbel_temperature', 1.0),
        lambda_consistency=mot_config.get('lambda_consistency', 0.05),
        consistency_temperature=mot_config.get('consistency_temperature', 2.0),
    )
    
    print(f"  MoT config: stacks={config.num_stacks}, top_k={config.top_k}")
    
    model = MixtureOfThoughts(expert_models, tokenizers, config)
    model = model.to(device)
    
    # Load weights
    model_state = model.state_dict()
    loaded = 0
    for name, param in checkpoint['model_state_dict'].items():
        if name in model_state and model_state[name].shape == param.shape:
            model_state[name] = param
            loaded += 1
    
    model.load_state_dict(model_state, strict=False)
    print(f"  Loaded {loaded} parameter tensors")
    print(f"  Checkpoint step: {checkpoint.get('step', 'N/A')}")
    
    return model


@torch.no_grad()
def evaluate_on_babeltower(model, samples: List[BabelTowerSample], args) -> Tuple[Dict, List]:
    """Run evaluation on BabelTower samples."""
    model.eval()
    
    predictions = []
    references = []
    all_results = []
    expert_usage = {}
    
    use_mot = args.use_mot_generate and not args.no_mot_generate
    max_samples = args.num_samples or len(samples)
    
    print(f"\nEvaluating on {min(max_samples, len(samples))} samples")
    print(f"  Generation mode: {'full MoT' if use_mot else 'fast'}")
    print(f"  Max new tokens: {args.max_new_tokens}")
    
    prompt_template = "### Translate C++ to CUDA:\n{source}\n### CUDA:\n"
    
    for i, sample in enumerate(tqdm(samples[:max_samples], desc="Evaluating")):
        try:
            generated_cuda, primary_idx = model.translate(
                source_text=sample.cpp_code,
                prompt_template=prompt_template,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                use_mot_generate=use_mot,
            )
            
            expert_usage[primary_idx] = expert_usage.get(primary_idx, 0) + 1
            predictions.append(generated_cuda)
            references.append(sample.cuda_code)
            
            all_results.append({
                'index': sample.index,
                'cpp_code': sample.cpp_code,
                'reference_cuda': sample.cuda_code,
                'generated_cuda': generated_cuda,
                'primary_expert': primary_idx,
            })
            
        except Exception as e:
            print(f"\n  Error at sample {i}: {e}")
            predictions.append("")
            references.append(sample.cuda_code)
            all_results.append({
                'index': sample.index,
                'cpp_code': sample.cpp_code,
                'reference_cuda': sample.cuda_code,
                'generated_cuda': "",
                'error': str(e),
            })
    
    # Compute metrics
    evaluator = BabelTowerEvaluator(check_compilation=args.check_compilation)
    metrics = evaluator.evaluate_all(predictions, references)
    metrics['expert_usage'] = expert_usage
    metrics['generation_mode'] = 'mot' if use_mot else 'fast'
    
    return metrics, all_results


def main():
    args = parse_args()
    
    print("\n" + "=" * 60)
    print("BabelTower Evaluation Configuration")
    print("=" * 60)
    print(f"  Dataset dir: {args.dataset_dir}")
    print(f"  Split: {args.split}")
    print(f"  Checkpoint: {args.checkpoint_path}")
    print(f"  Max new tokens: {args.max_new_tokens}")
    print(f"  Use MoT generate: {args.use_mot_generate and not args.no_mot_generate}")
    print(f"  Num samples: {args.num_samples or 'all'}")
    print(f"  Output dir: {args.output_dir}")
    print("=" * 60)
    
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Validate paths
    if not os.path.exists(args.dataset_dir):
        print(f"\nError: Dataset not found: {args.dataset_dir}")
        sys.exit(1)
    if not os.path.exists(args.checkpoint_path):
        print(f"\nError: Checkpoint not found: {args.checkpoint_path}")
        sys.exit(1)
    
    # Load dataset
    print("\n" + "=" * 60)
    print("Loading BabelTower Dataset")
    print("=" * 60)
    samples = load_babeltower_dataset(args.dataset_dir, args.split)
    print(f"  Loaded {len(samples)} samples from {args.split} set")
    
    # Load models
    expert_models, tokenizers = load_expert_models(args)
    
    # Load MoT
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_mot_from_checkpoint(args.checkpoint_path, expert_models, tokenizers, device)
    
    # Run evaluation
    print("\n" + "=" * 60)
    print("Running Evaluation")
    print("=" * 60)
    
    metrics, all_results = evaluate_on_babeltower(model, samples, args)
    
    # Print results
    evaluator = BabelTowerEvaluator()
    evaluator.print_results(metrics)
    
    if 'expert_usage' in metrics:
        print(f"Expert usage: {metrics['expert_usage']}")
    
    # Save results
    metrics_path = os.path.join(args.output_dir, f'babeltower_{args.split}_metrics.json')
    save_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float, str, list, dict))}
    save_metrics['checkpoint_path'] = args.checkpoint_path
    save_metrics['split'] = args.split
    with open(metrics_path, 'w') as f:
        json.dump(save_metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")
    
    if args.save_predictions:
        predictions_path = os.path.join(args.output_dir, f'babeltower_{args.split}_predictions.json')
        with open(predictions_path, 'w') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"Predictions saved to {predictions_path}")
    
    print("\n" + "=" * 60)
    print("Evaluation Complete")
    print("=" * 60)


if __name__ == '__main__':
    main()