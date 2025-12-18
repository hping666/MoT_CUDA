"""
Dataset loaders for benchmark datasets used in MoT experiments.
Supports MMLU, GSM8K, CMMLU, ARC-Challenge, HumanEval, and OOD datasets.
"""

import torch
from torch.utils.data import Dataset, ConcatDataset
from datasets import load_dataset, concatenate_datasets
from typing import List, Dict, Optional, Union, Any
import json
import random
import re
from pathlib import Path


class BenchmarkDataset(Dataset):
    """Base dataset class for benchmark tasks."""
    
    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        max_length: int = 512,
        task_type: str = 'generation'
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_type = task_type
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format input based on task type
        if self.task_type == 'multiple_choice':
            text = self._format_multiple_choice(item)
        elif self.task_type == 'math':
            text = self._format_math_problem(item)
        elif self.task_type == 'code':
            text = self._format_code_problem(item)
        else:
            text = self._format_generation(item)
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Create labels (for language modeling)
        labels = encoding['input_ids'].clone()
        
        # Mask padding tokens in labels
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': labels.squeeze(),
            'task_type': self.task_type,
            'dataset_name': item.get('dataset_name', 'unknown')
        }
    
    def _format_multiple_choice(self, item: Dict) -> str:
        """Format multiple choice question."""
        question = item['question']
        options = item.get('options', [])
        answer = item.get('answer', '')
        
        text = f"Question: {question}\n"
        if options:
            text += "Options:\n"
            for i, opt in enumerate(options):
                text += f"{chr(65+i)}. {opt}\n"
        text += f"Answer: {answer}"
        
        return text
    
    def _format_math_problem(self, item: Dict) -> str:
        """Format math problem."""
        question = item['question']
        answer = item.get('answer', '')
        solution = item.get('solution', '')
        
        text = f"Problem: {question}\n"
        if solution:
            text += f"Solution: {solution}\n"
        text += f"Answer: {answer}"
        
        return text
    
    def _format_code_problem(self, item: Dict) -> str:
        """Format code generation problem."""
        prompt = item.get('prompt', item.get('question', ''))
        code = item.get('code', item.get('answer', ''))
        
        text = f"Task: {prompt}\n"
        text += f"Code:\n```python\n{code}\n```"
        
        return text
    
    def _format_generation(self, item: Dict) -> str:
        """Format general generation task."""
        question = item.get('question', item.get('prompt', ''))
        answer = item.get('answer', item.get('response', ''))
        
        return f"{question} {answer}"


class MMLUDataset(BenchmarkDataset):
    """MMLU dataset loader."""
    
    @staticmethod
    def load_data(split: str = 'train') -> List[Dict]:
        """Load MMLU dataset."""
        dataset = load_dataset('cais/mmlu', 'all', split=split)
        
        data = []
        for item in dataset:
            choices = [item['choices'][i] for i in range(len(item['choices']))]
            answer_idx = item['answer']
            
            data.append({
                'question': item['question'],
                'options': choices,
                'answer': choices[answer_idx] if answer_idx < len(choices) else '',
                'subject': item['subject'],
                'dataset_name': 'mmlu'
            })
        
        return data


class GSM8KDataset(BenchmarkDataset):
    """GSM8K dataset loader."""
    
    @staticmethod
    def load_data(split: str = 'train') -> List[Dict]:
        """Load GSM8K dataset."""
        dataset = load_dataset('gsm8k', 'main', split=split)
        
        data = []
        for item in dataset:
            # Extract numerical answer from the solution
            answer_match = re.search(r'#### ([\d,]+)', item['answer'])
            answer = answer_match.group(1) if answer_match else ''
            
            data.append({
                'question': item['question'],
                'solution': item['answer'].split('####')[0].strip(),
                'answer': answer,
                'dataset_name': 'gsm8k'
            })
        
        return data


class CMMLUDataset(BenchmarkDataset):
    """CMMLU (Chinese MMLU) dataset loader."""
    
    @staticmethod
    def load_data(split: str = 'train') -> List[Dict]:
        """Load CMMLU dataset."""
        try:
            dataset = load_dataset('haonan-li/cmmlu', 'all', split=split)
        except:
            # Fallback if dataset not available
            print("Warning: CMMLU dataset not available, using placeholder data")
            return []
        
        data = []
        for item in dataset:
            choices = [item[f'choice_{i}'] for i in range(4)]
            answer_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
            answer_idx = answer_map.get(item['answer'], 0)
            
            data.append({
                'question': item['question'],
                'options': choices,
                'answer': choices[answer_idx],
                'subject': item.get('subject', ''),
                'dataset_name': 'cmmlu'
            })
        
        return data


class ARCChallengeDataset(BenchmarkDataset):
    """ARC-Challenge dataset loader."""
    
    @staticmethod
    def load_data(split: str = 'train') -> List[Dict]:
        """Load ARC-Challenge dataset."""
        dataset = load_dataset('ai2_arc', 'ARC-Challenge', split=split)
        
        data = []
        for item in dataset:
            choices = item['choices']['text']
            labels = item['choices']['label']
            answer_key = item['answerKey']
            
            # Find the correct answer
            answer_idx = labels.index(answer_key) if answer_key in labels else 0
            
            data.append({
                'question': item['question'],
                'options': choices,
                'answer': choices[answer_idx],
                'dataset_name': 'arc_challenge'
            })
        
        return data


class HumanEvalDataset(BenchmarkDataset):
    """HumanEval dataset loader."""
    
    @staticmethod
    def load_data(split: str = 'test') -> List[Dict]:
        """Load HumanEval dataset."""
        dataset = load_dataset('openai_humaneval', split='test')  # HumanEval only has test split
        
        data = []
        for item in dataset:
            data.append({
                'prompt': item['prompt'],
                'code': item['canonical_solution'],
                'test': item['test'],
                'task_id': item['task_id'],
                'dataset_name': 'humaneval'
            })
        
        return data


class PreAlgebraDataset(BenchmarkDataset):
    """PreAlgebra dataset loader (from MATH dataset)."""
    
    @staticmethod
    def load_data(split: str = 'train') -> List[Dict]:
        """Load PreAlgebra dataset."""
        try:
            dataset = load_dataset('hendrycks/math', 'prealgebra', split=split)
        except:
            # Use MATH dataset as fallback
            dataset = load_dataset('competition_math', split=split)
            # Filter for prealgebra problems
            dataset = dataset.filter(lambda x: 'prealgebra' in x.get('type', '').lower())
        
        data = []
        for item in dataset:
            data.append({
                'question': item['problem'],
                'solution': item.get('solution', ''),
                'answer': item.get('answer', ''),
                'level': item.get('level', ''),
                'dataset_name': 'prealgebra'
            })
        
        return data


class MBPPDataset(BenchmarkDataset):
    """MBPP (Mostly Basic Python Problems) dataset loader."""
    
    @staticmethod
    def load_data(split: str = 'train') -> List[Dict]:
        """Load MBPP dataset."""
        # MBPP uses different split names
        split_map = {'train': 'train', 'validation': 'validation', 'test': 'test'}
        split = split_map.get(split, 'train')
        
        dataset = load_dataset('mbpp', split=split)
        
        data = []
        for item in dataset:
            data.append({
                'prompt': item['text'],
                'code': item['code'],
                'test_list': item['test_list'],
                'task_id': item['task_id'],
                'dataset_name': 'mbpp'
            })
        
        return data


class CEvalDataset(BenchmarkDataset):
    """C-Eval dataset loader."""
    
    @staticmethod
    def load_data(split: str = 'train') -> List[Dict]:
        """Load C-Eval dataset."""
        try:
            dataset = load_dataset('ceval/ceval-exam', 'all', split=split)
        except:
            print("Warning: C-Eval dataset not available, using placeholder data")
            return []
        
        data = []
        for item in dataset:
            choices = [item[f'choice_{i}'] for i in ['A', 'B', 'C', 'D']]
            answer_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
            answer_idx = answer_map.get(item['answer'], 0)
            
            data.append({
                'question': item['question'],
                'options': choices,
                'answer': choices[answer_idx],
                'subject': item.get('subject', ''),
                'dataset_name': 'c_eval'
            })
        
        return data


def load_benchmark_dataset(
    dataset_names: Union[str, List[str]],
    tokenizer,
    max_length: int = 512,
    split: str = 'train',
    sample_size: Optional[int] = None
) -> Dataset:
    """
    Load and combine multiple benchmark datasets.
    
    Args:
        dataset_names: Name(s) of datasets to load
        tokenizer: Tokenizer to use for preprocessing
        max_length: Maximum sequence length
        split: Dataset split ('train', 'validation', 'test')
        sample_size: If specified, sample this many examples per dataset
    
    Returns:
        Combined dataset
    """
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]
    
    # Dataset loaders mapping
    dataset_loaders = {
        'mmlu': (MMLUDataset, 'multiple_choice'),
        'gsm8k': (GSM8KDataset, 'math'),
        'cmmlu': (CMMLUDataset, 'multiple_choice'),
        'arc_challenge': (ARCChallengeDataset, 'multiple_choice'),
        'humaneval': (HumanEvalDataset, 'code'),
        'prealgebra': (PreAlgebraDataset, 'math'),
        'mbpp': (MBPPDataset, 'code'),
        'c_eval': (CEvalDataset, 'multiple_choice')
    }
    
    all_datasets = []
    
    for dataset_name in dataset_names:
        if dataset_name not in dataset_loaders:
            print(f"Warning: Unknown dataset {dataset_name}, skipping...")
            continue
        
        dataset_class, task_type = dataset_loaders[dataset_name]
        
        # Load data
        try:
            data = dataset_class.load_data(split)
            
            # Sample if requested
            if sample_size and len(data) > sample_size:
                data = random.sample(data, sample_size)
            
            # Create dataset
            dataset = dataset_class(
                data=data,
                tokenizer=tokenizer,
                max_length=max_length,
                task_type=task_type
            )
            
            all_datasets.append(dataset)
            print(f"Loaded {dataset_name}: {len(dataset)} examples")
            
        except Exception as e:
            print(f"Error loading {dataset_name}: {e}")
            continue
    
    # Combine all datasets
    if not all_datasets:
        raise ValueError("No datasets were successfully loaded")
    
    if len(all_datasets) == 1:
        return all_datasets[0]
    
    return ConcatDataset(all_datasets)


class DataCollatorForMoT:
    """Data collator for MoT training."""
    
    def __init__(self, tokenizer, pad_to_multiple_of: Optional[int] = None):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate batch of features."""
        # Extract tensors
        batch = {
            'input_ids': torch.stack([f['input_ids'] for f in features]),
            'attention_mask': torch.stack([f['attention_mask'] for f in features]),
            'labels': torch.stack([f['labels'] for f in features])
        }
        
        # Add metadata if present
        if 'task_type' in features[0]:
            batch['task_types'] = [f['task_type'] for f in features]
        
        if 'dataset_name' in features[0]:
            batch['dataset_names'] = [f['dataset_name'] for f in features]
        
        return batch


def create_train_val_split(
    dataset: Dataset,
    val_ratio: float = 0.1,
    seed: int = 42
) -> tuple:
    """
    Split dataset into train and validation sets.
    
    Args:
        dataset: Dataset to split
        val_ratio: Ratio of validation data
        seed: Random seed
    
    Returns:
        train_dataset, val_dataset
    """
    # Set random seed
    random.seed(seed)
    
    # Calculate split sizes
    total_size = len(dataset)
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size
    
    # Random split
    indices = list(range(total_size))
    random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create subset datasets
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    return train_dataset, val_dataset


# Evaluation utilities
def evaluate_predictions(
    predictions: List[str],
    references: List[str],
    task_type: str
) -> Dict[str, float]:
    """
    Evaluate predictions based on task type.
    
    Args:
        predictions: Model predictions
        references: Ground truth answers
        task_type: Type of task
    
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    if task_type == 'multiple_choice':
        # Exact match for multiple choice
        correct = sum(p.strip() == r.strip() for p, r in zip(predictions, references))
        metrics['accuracy'] = correct / len(predictions)
        
    elif task_type == 'math':
        # Extract numerical answers and compare
        def extract_number(text):
            numbers = re.findall(r'[-+]?\d*\.?\d+', text)
            return numbers[-1] if numbers else ''
        
        pred_numbers = [extract_number(p) for p in predictions]
        ref_numbers = [extract_number(r) for r in references]
        correct = sum(p == r for p, r in zip(pred_numbers, ref_numbers))
        metrics['accuracy'] = correct / len(predictions)
        
    elif task_type == 'code':
        # For code, we'd need to execute and test
        # This is a placeholder - actual implementation would run tests
        metrics['pass_rate'] = 0.0  # Placeholder
        
    else:
        # General text generation - could use BLEU, ROUGE, etc.
        metrics['exact_match'] = sum(
            p.strip() == r.strip() for p, r in zip(predictions, references)
        ) / len(predictions)
    
    return metrics
