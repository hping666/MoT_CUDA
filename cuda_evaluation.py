"""
Evaluation metrics for CUDA code generation.
Computes BLEU, chrF, ROUGE-L, Exact Match, and Edit Similarity.
"""

import re
from typing import List, Dict, Any, Optional
from collections import Counter
import math


def normalize_code(code: str) -> str:
    """
    Normalize code for comparison.
    
    - Remove extra whitespace
    - Normalize line endings
    - Strip leading/trailing whitespace
    """
    # Replace multiple spaces/tabs with single space
    code = re.sub(r'[ \t]+', ' ', code)
    # Normalize line endings
    code = re.sub(r'\r\n|\r', '\n', code)
    # Remove empty lines
    code = re.sub(r'\n\s*\n', '\n', code)
    # Strip each line
    lines = [line.strip() for line in code.split('\n')]
    code = '\n'.join(line for line in lines if line)
    return code.strip()


def tokenize_code(code: str) -> List[str]:
    """
    Tokenize code into tokens for BLEU computation.
    Handles code-specific tokenization.
    """
    # Normalize first
    code = normalize_code(code)
    
    # Split on whitespace and punctuation while keeping punctuation
    tokens = re.findall(r'\w+|[^\w\s]', code)
    
    return tokens


class CudaCodeEvaluator:
    """Evaluator for CUDA code generation quality."""
    
    def __init__(self):
        """Initialize the evaluator."""
        pass
    
    def compute_bleu(
        self,
        predictions: List[str],
        references: List[str],
        max_n: int = 4,
        weights: Optional[List[float]] = None
    ) -> float:
        """
        Compute BLEU score.
        
        Args:
            predictions: List of predicted CUDA codes
            references: List of reference CUDA codes
            max_n: Maximum n-gram order (default: 4 for BLEU-4)
            weights: Weights for each n-gram order
            
        Returns:
            BLEU score (0-100)
        """
        if weights is None:
            weights = [1.0 / max_n] * max_n
        
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length")
        
        if not predictions:
            return 0.0
        
        # Collect n-gram statistics
        clipped_counts = [0] * max_n
        total_counts = [0] * max_n
        ref_lengths = 0
        pred_lengths = 0
        
        for pred, ref in zip(predictions, references):
            pred_tokens = tokenize_code(pred)
            ref_tokens = tokenize_code(ref)
            
            pred_lengths += len(pred_tokens)
            ref_lengths += len(ref_tokens)
            
            # Compute n-gram matches
            for n in range(1, max_n + 1):
                pred_ngrams = self._get_ngrams(pred_tokens, n)
                ref_ngrams = self._get_ngrams(ref_tokens, n)
                
                # Count clipped matches
                for ngram, count in pred_ngrams.items():
                    clipped_counts[n-1] += min(count, ref_ngrams.get(ngram, 0))
                    total_counts[n-1] += count
        
        # Compute modified precision for each n
        precisions = []
        for n in range(max_n):
            if total_counts[n] > 0:
                precisions.append(clipped_counts[n] / total_counts[n])
            else:
                precisions.append(0.0)
        
        # Compute geometric mean of precisions
        if min(precisions) > 0:
            log_precision = sum(w * math.log(p) for w, p in zip(weights, precisions))
            geo_mean = math.exp(log_precision)
        else:
            geo_mean = 0.0
        
        # Compute brevity penalty
        if pred_lengths > 0:
            bp = math.exp(min(0, 1 - ref_lengths / pred_lengths))
        else:
            bp = 0.0
        
        bleu = bp * geo_mean * 100  # Scale to 0-100
        return bleu
    
    def _get_ngrams(self, tokens: List[str], n: int) -> Counter:
        """Get n-gram counts from token list."""
        ngrams = Counter()
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngrams[ngram] += 1
        return ngrams
    
    def compute_chrf(
        self,
        predictions: List[str],
        references: List[str],
        char_order: int = 6,
        beta: float = 2.0
    ) -> float:
        """
        Compute chrF score (character-level F-score).
        
        Args:
            predictions: List of predicted CUDA codes
            references: List of reference CUDA codes
            char_order: Maximum character n-gram order
            beta: Parameter for F-score (beta > 1 favors recall)
            
        Returns:
            chrF score (0-100)
        """
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length")
        
        if not predictions:
            return 0.0
        
        total_precision = 0.0
        total_recall = 0.0
        count = 0
        
        for pred, ref in zip(predictions, references):
            pred_normalized = normalize_code(pred)
            ref_normalized = normalize_code(ref)
            
            # Compute character n-gram F-scores
            precision_sum = 0.0
            recall_sum = 0.0
            
            for n in range(1, char_order + 1):
                pred_ngrams = self._get_char_ngrams(pred_normalized, n)
                ref_ngrams = self._get_char_ngrams(ref_normalized, n)
                
                # Compute precision and recall
                matches = sum((pred_ngrams & ref_ngrams).values())
                pred_total = sum(pred_ngrams.values())
                ref_total = sum(ref_ngrams.values())
                
                if pred_total > 0:
                    precision_sum += matches / pred_total
                if ref_total > 0:
                    recall_sum += matches / ref_total
            
            # Average over n-gram orders
            if char_order > 0:
                avg_precision = precision_sum / char_order
                avg_recall = recall_sum / char_order
                
                # Compute F-score
                if avg_precision + avg_recall > 0:
                    f_score = (1 + beta**2) * avg_precision * avg_recall / (beta**2 * avg_precision + avg_recall)
                else:
                    f_score = 0.0
                
                total_precision += avg_precision
                total_recall += avg_recall
                count += 1
        
        if count == 0:
            return 0.0
        
        # Compute corpus-level chrF
        avg_precision = total_precision / count
        avg_recall = total_recall / count
        
        if avg_precision + avg_recall > 0:
            chrf = (1 + beta**2) * avg_precision * avg_recall / (beta**2 * avg_precision + avg_recall)
        else:
            chrf = 0.0
        
        return chrf * 100  # Scale to 0-100
    
    def _get_char_ngrams(self, text: str, n: int) -> Counter:
        """Get character n-gram counts."""
        ngrams = Counter()
        for i in range(len(text) - n + 1):
            ngram = text[i:i+n]
            ngrams[ngram] += 1
        return ngrams
    
    def compute_rouge_l(
        self,
        predictions: List[str],
        references: List[str]
    ) -> float:
        """
        Compute ROUGE-L score (Longest Common Subsequence based).
        
        Args:
            predictions: List of predicted CUDA codes
            references: List of reference CUDA codes
            
        Returns:
            ROUGE-L F1 score (0-100)
        """
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length")
        
        if not predictions:
            return 0.0
        
        total_f1 = 0.0
        
        for pred, ref in zip(predictions, references):
            pred_tokens = tokenize_code(pred)
            ref_tokens = tokenize_code(ref)
            
            # Compute LCS length
            lcs_length = self._lcs_length(pred_tokens, ref_tokens)
            
            # Compute precision and recall
            if len(pred_tokens) > 0:
                precision = lcs_length / len(pred_tokens)
            else:
                precision = 0.0
            
            if len(ref_tokens) > 0:
                recall = lcs_length / len(ref_tokens)
            else:
                recall = 0.0
            
            # Compute F1
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0
            
            total_f1 += f1
        
        return (total_f1 / len(predictions)) * 100  # Scale to 0-100
    
    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """Compute length of Longest Common Subsequence."""
        m, n = len(seq1), len(seq2)
        
        # Use space-optimized DP
        if m < n:
            seq1, seq2 = seq2, seq1
            m, n = n, m
        
        prev = [0] * (n + 1)
        curr = [0] * (n + 1)
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    curr[j] = prev[j-1] + 1
                else:
                    curr[j] = max(prev[j], curr[j-1])
            prev, curr = curr, prev
        
        return prev[n]
    
    def compute_exact_match(
        self,
        predictions: List[str],
        references: List[str],
        normalize: bool = True
    ) -> float:
        """
        Compute Exact Match rate.
        
        Args:
            predictions: List of predicted CUDA codes
            references: List of reference CUDA codes
            normalize: Whether to normalize code before comparison
            
        Returns:
            Exact match rate (0-100)
        """
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length")
        
        if not predictions:
            return 0.0
        
        matches = 0
        for pred, ref in zip(predictions, references):
            if normalize:
                pred = normalize_code(pred)
                ref = normalize_code(ref)
            
            if pred == ref:
                matches += 1
        
        return (matches / len(predictions)) * 100  # Scale to 0-100
    
    def compute_edit_similarity(
        self,
        predictions: List[str],
        references: List[str]
    ) -> float:
        """
        Compute Edit Similarity (1 - normalized edit distance).
        
        Args:
            predictions: List of predicted CUDA codes
            references: List of reference CUDA codes
            
        Returns:
            Average edit similarity (0-100)
        """
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length")
        
        if not predictions:
            return 0.0
        
        total_similarity = 0.0
        
        for pred, ref in zip(predictions, references):
            pred_normalized = normalize_code(pred)
            ref_normalized = normalize_code(ref)
            
            # Compute Levenshtein distance
            distance = self._levenshtein_distance(pred_normalized, ref_normalized)
            
            # Normalize by max length
            max_len = max(len(pred_normalized), len(ref_normalized))
            if max_len > 0:
                similarity = 1 - (distance / max_len)
            else:
                similarity = 1.0
            
            total_similarity += similarity
        
        return (total_similarity / len(predictions)) * 100  # Scale to 0-100
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Compute Levenshtein (edit) distance between two strings."""
        if len(s1) < len(s2):
            s1, s2 = s2, s1
        
        if len(s2) == 0:
            return len(s1)
        
        prev_row = list(range(len(s2) + 1))
        
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                # Cost is 0 if characters match, 1 otherwise
                insertions = prev_row[j + 1] + 1
                deletions = curr_row[j] + 1
                substitutions = prev_row[j] + (0 if c1 == c2 else 1)
                curr_row.append(min(insertions, deletions, substitutions))
            prev_row = curr_row
        
        return prev_row[-1]
    
    def compute_code_compilability(
        self,
        predictions: List[str]
    ) -> float:
        """
        Check if generated CUDA code has basic syntax validity.
        This is a simple heuristic check, not actual compilation.
        
        Args:
            predictions: List of predicted CUDA codes
            
        Returns:
            Rate of syntactically plausible code (0-100)
        """
        if not predictions:
            return 0.0
        
        valid_count = 0
        
        for pred in predictions:
            if self._basic_cuda_syntax_check(pred):
                valid_count += 1
        
        return (valid_count / len(predictions)) * 100
    
    def _basic_cuda_syntax_check(self, code: str) -> bool:
        """
        Perform basic CUDA syntax checks.
        
        Checks for:
        - Balanced braces
        - Presence of __global__ or __device__ keyword
        - Basic structure
        """
        # Check for CUDA kernel indicators
        has_cuda_keyword = '__global__' in code or '__device__' in code
        
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
        
        # Check for function-like structure
        has_function = re.search(r'\w+\s*\([^)]*\)\s*{', code) is not None
        
        return has_cuda_keyword and balanced and has_function
    
    def evaluate_all(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        Compute all evaluation metrics.
        
        Args:
            predictions: List of predicted CUDA codes
            references: List of reference CUDA codes
            
        Returns:
            Dictionary with all metrics
        """
        results = {
            'bleu': self.compute_bleu(predictions, references),
            'chrf': self.compute_chrf(predictions, references),
            'rouge_l': self.compute_rouge_l(predictions, references),
            'exact_match': self.compute_exact_match(predictions, references),
            'edit_similarity': self.compute_edit_similarity(predictions, references),
            'syntax_validity': self.compute_code_compilability(predictions),
            'num_samples': len(predictions)
        }
        
        return results
    
    def print_results(self, results: Dict[str, float]) -> None:
        """Print evaluation results in a formatted way."""
        print("\n" + "=" * 50)
        print("CUDA Code Generation Evaluation Results")
        print("=" * 50)
        print(f"  Samples evaluated: {results.get('num_samples', 'N/A')}")
        print("-" * 50)
        print(f"  BLEU-4:           {results.get('bleu', 0):.2f}")
        print(f"  chrF:             {results.get('chrf', 0):.2f}")
        print(f"  ROUGE-L:          {results.get('rouge_l', 0):.2f}")
        print(f"  Exact Match:      {results.get('exact_match', 0):.2f}%")
        print(f"  Edit Similarity:  {results.get('edit_similarity', 0):.2f}%")
        print(f"  Syntax Validity:  {results.get('syntax_validity', 0):.2f}%")
        print("=" * 50 + "\n")


def evaluate_generation(
    predictions: List[str],
    references: List[str],
    print_results: bool = True
) -> Dict[str, float]:
    """
    Convenience function to evaluate CUDA code generation.
    
    Args:
        predictions: List of predicted CUDA codes
        references: List of reference CUDA codes
        print_results: Whether to print results
        
    Returns:
        Dictionary with all metrics
    """
    evaluator = CudaCodeEvaluator()
    results = evaluator.evaluate_all(predictions, references)
    
    if print_results:
        evaluator.print_results(results)
    
    return results


if __name__ == '__main__':
    # Test the evaluator
    predictions = [
        "__global__ void test(int *a) {\n  int i = blockIdx.x * blockDim.x + threadIdx.x;\n  a[i] = i;\n}",
        "__global__ void add(float *a, float *b) {\n  int idx = threadIdx.x;\n  a[idx] += b[idx];\n}"
    ]
    
    references = [
        "__global__ void test(int *a) {\n  int i = blockIdx.x * blockDim.x + threadIdx.x;\n  a[i] = i;\n}",
        "__global__ void add(float *a, float *b, int n) {\n  int idx = blockIdx.x * blockDim.x + threadIdx.x;\n  if (idx < n) a[idx] += b[idx];\n}"
    ]
    
    print("Testing CUDA Code Evaluator")
    print("-" * 40)
    
    results = evaluate_generation(predictions, references)