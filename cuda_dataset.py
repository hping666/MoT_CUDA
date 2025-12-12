"""
Dataset loader for C++ to CUDA code translation task.
Supports configurable train/test split with overlap control.

FIXES:
- Added raw_text preservation in batch for multi-tokenizer support
- DataCollator now passes through raw text for expert-specific encoding
"""

import json
import random
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path


class CppCudaDataset(Dataset):
    """Dataset for C++ to CUDA code translation."""
    
    # Prompt template for the translation task
    PROMPT_TEMPLATE = "### Translate C++ to CUDA:\n{cpp}\n### CUDA:\n"
    
    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        max_length: int = 1024,
        is_train: bool = True
    ):
        """
        Initialize the dataset.
        
        Args:
            data: List of dictionaries with 'cpp' and 'generated_cuda' keys
            tokenizer: Tokenizer for encoding text (primary tokenizer)
            max_length: Maximum sequence length
            is_train: Whether this is training data (includes target) or inference
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_train = is_train
        
        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        cpp_code = item['cpp']
        cuda_code = item['generated_cuda']
        
        # Format the prompt
        prompt = self.PROMPT_TEMPLATE.format(cpp=cpp_code)
        
        if self.is_train:
            # For training: include both prompt and target
            full_text = prompt + cuda_code + self.tokenizer.eos_token
            
            # Tokenize the full sequence
            encoding = self.tokenizer(
                full_text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].squeeze(0)
            attention_mask = encoding['attention_mask'].squeeze(0)
            
            # Create labels: mask the prompt part (set to -100)
            labels = input_ids.clone()
            
            # Find where the prompt ends (after "### CUDA:\n")
            prompt_encoding = self.tokenizer(
                prompt,
                add_special_tokens=False,
                return_tensors='pt'
            )
            prompt_length = prompt_encoding['input_ids'].shape[1]
            
            # Mask prompt tokens in labels
            labels[:prompt_length] = -100
            
            # Also mask padding tokens
            labels[attention_mask == 0] = -100
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'cpp_code': cpp_code,
                'cuda_code': cuda_code,
                # NEW: Add raw texts for multi-tokenizer support
                'raw_prompt': prompt,
                'raw_full_text': full_text,
                'prompt_length': prompt_length,
            }
        else:
            # For inference: only include the prompt
            encoding = self.tokenizer(
                prompt,
                max_length=self.max_length // 2,  # Leave room for generation
                padding=False,
                truncation=True,
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
                'cpp_code': cpp_code,
                'cuda_code': cuda_code,  # Reference for evaluation
                # NEW: Add raw texts for multi-tokenizer support
                'raw_prompt': prompt,
                'raw_full_text': prompt,  # For inference, full_text is just the prompt
            }
    
    @staticmethod
    def format_prompt(cpp_code: str) -> str:
        """Format a C++ code snippet into a translation prompt."""
        return CppCudaDataset.PROMPT_TEMPLATE.format(cpp=cpp_code)


def load_cpp_cuda_data(
    filepath: str,
    train_ratio: float = 0.8,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """
    Load and split C++ to CUDA dataset.
    
    Args:
        filepath: Path to the JSONL file
        train_ratio: Ratio of data to use for training (0.0 to 1.0+)
        test_ratio: Ratio of data to use for testing (0.0 to 1.0+)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_data, test_data)
        
    Note:
        - If train_ratio + test_ratio <= 1.0: No overlap between train and test
        - If train_ratio + test_ratio > 1.0: Overlap is allowed
    """
    # Set random seed
    random.seed(seed)
    
    # Load all data
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset file not found: {filepath}")
    
    all_data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    item = json.loads(line)
                    # Validate required fields
                    if 'cpp' in item and 'generated_cuda' in item:
                        all_data.append(item)
                except json.JSONDecodeError:
                    continue
    
    if not all_data:
        raise ValueError(f"No valid data found in {filepath}")
    
    n = len(all_data)
    print(f"Loaded {n} samples from {filepath}")
    
    # Shuffle data
    shuffled_data = all_data.copy()
    random.shuffle(shuffled_data)
    
    # Determine split strategy based on ratios
    total_ratio = train_ratio + test_ratio
    
    if total_ratio <= 1.0:
        # No overlap: sequential split
        train_end = int(n * train_ratio)
        test_end = train_end + int(n * test_ratio)
        
        train_data = shuffled_data[:train_end]
        test_data = shuffled_data[train_end:test_end]
        
        print(f"Split mode: No overlap (train_ratio + test_ratio = {total_ratio:.2f} <= 1.0)")
    else:
        # Allow overlap: independent sampling
        train_size = min(int(n * train_ratio), n)
        test_size = min(int(n * test_ratio), n)
        
        # For training, use all available if ratio > 1
        if train_ratio >= 1.0:
            train_data = shuffled_data.copy()
            # If we need more, sample with replacement
            if int(n * train_ratio) > n:
                extra_needed = int(n * train_ratio) - n
                train_data.extend(random.choices(all_data, k=extra_needed))
        else:
            train_data = shuffled_data[:train_size]
        
        # For testing, sample independently
        test_data = random.sample(all_data, test_size)
        
        print(f"Split mode: With overlap (train_ratio + test_ratio = {total_ratio:.2f} > 1.0)")
    
    print(f"Train samples: {len(train_data)} ({len(train_data)/n*100:.1f}%)")
    print(f"Test samples: {len(test_data)} ({len(test_data)/n*100:.1f}%)")
    
    return train_data, test_data


class DataCollatorForCppCuda:
    """Data collator for C++ to CUDA translation task."""
    
    def __init__(self, tokenizer, padding: bool = True, max_length: Optional[int] = None):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
        
        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate batch of features."""
        batch = {}
        
        # Handle input_ids
        if 'input_ids' in features[0]:
            input_ids = [f['input_ids'] for f in features]
            
            # Pad sequences
            if self.padding:
                max_len = max(len(ids) for ids in input_ids)
                if self.max_length:
                    max_len = min(max_len, self.max_length)
                
                padded_input_ids = []
                padded_attention_mask = []
                
                for i, ids in enumerate(input_ids):
                    if len(ids) < max_len:
                        padding_length = max_len - len(ids)
                        ids = torch.cat([
                            ids,
                            torch.full((padding_length,), self.tokenizer.pad_token_id, dtype=ids.dtype)
                        ])
                        mask = torch.cat([
                            features[i]['attention_mask'],
                            torch.zeros(padding_length, dtype=features[i]['attention_mask'].dtype)
                        ])
                    else:
                        ids = ids[:max_len]
                        mask = features[i]['attention_mask'][:max_len]
                    
                    padded_input_ids.append(ids)
                    padded_attention_mask.append(mask)
                
                batch['input_ids'] = torch.stack(padded_input_ids)
                batch['attention_mask'] = torch.stack(padded_attention_mask)
            else:
                batch['input_ids'] = torch.stack(input_ids)
                batch['attention_mask'] = torch.stack([f['attention_mask'] for f in features])
        
        # Handle labels
        if 'labels' in features[0]:
            labels = [f['labels'] for f in features]
            
            if self.padding:
                max_len = batch['input_ids'].shape[1]
                padded_labels = []
                
                for lbl in labels:
                    if len(lbl) < max_len:
                        padding_length = max_len - len(lbl)
                        lbl = torch.cat([
                            lbl,
                            torch.full((padding_length,), -100, dtype=lbl.dtype)
                        ])
                    else:
                        lbl = lbl[:max_len]
                    padded_labels.append(lbl)
                
                batch['labels'] = torch.stack(padded_labels)
            else:
                batch['labels'] = torch.stack(labels)
        
        # Keep metadata as lists
        if 'cpp_code' in features[0]:
            batch['cpp_code'] = [f['cpp_code'] for f in features]
        if 'cuda_code' in features[0]:
            batch['cuda_code'] = [f['cuda_code'] for f in features]
        
        # NEW: Keep raw texts for multi-tokenizer support
        if 'raw_prompt' in features[0]:
            batch['raw_prompt'] = [f['raw_prompt'] for f in features]
        if 'raw_full_text' in features[0]:
            batch['raw_full_text'] = [f['raw_full_text'] for f in features]
        if 'prompt_length' in features[0]:
            batch['prompt_length'] = [f['prompt_length'] for f in features]
        
        return batch


def create_dataloaders(
    train_data: List[Dict],
    test_data: List[Dict],
    tokenizer,
    batch_size: int = 1,
    max_length: int = 1024,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """
    Create DataLoaders for training and testing.
    
    Args:
        train_data: Training data list
        test_data: Test data list
        tokenizer: Tokenizer to use
        batch_size: Batch size
        max_length: Maximum sequence length
        num_workers: Number of worker processes
        
    Returns:
        Tuple of (train_dataloader, test_dataloader)
    """
    # Create datasets
    train_dataset = CppCudaDataset(
        data=train_data,
        tokenizer=tokenizer,
        max_length=max_length,
        is_train=True
    )
    
    test_dataset = CppCudaDataset(
        data=test_data,
        tokenizer=tokenizer,
        max_length=max_length,
        is_train=False  # For evaluation, we generate and compare
    )
    
    # Create data collator
    collator = DataCollatorForCppCuda(tokenizer, padding=True, max_length=max_length)
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,  # Generate one at a time for evaluation
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_dataloader, test_dataloader


# Utility functions
def get_dataset_statistics(filepath: str) -> Dict[str, Any]:
    """
    Get statistics about the dataset.
    
    Args:
        filepath: Path to the JSONL file
        
    Returns:
        Dictionary with dataset statistics
    """
    cpp_lengths = []
    cuda_lengths = []
    cpp_tokens = []
    cuda_tokens = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    item = json.loads(line)
                    if 'cpp' in item and 'generated_cuda' in item:
                        cpp_lengths.append(len(item['cpp']))
                        cuda_lengths.append(len(item['generated_cuda']))
                        cpp_tokens.append(len(item['cpp'].split()))
                        cuda_tokens.append(len(item['generated_cuda'].split()))
                except json.JSONDecodeError:
                    continue
    
    if not cpp_lengths:
        return {'error': 'No valid data found'}
    
    return {
        'total_samples': len(cpp_lengths),
        'cpp_char_length': {
            'min': min(cpp_lengths),
            'max': max(cpp_lengths),
            'mean': sum(cpp_lengths) / len(cpp_lengths),
        },
        'cuda_char_length': {
            'min': min(cuda_lengths),
            'max': max(cuda_lengths),
            'mean': sum(cuda_lengths) / len(cuda_lengths),
        },
        'cpp_token_count': {
            'min': min(cpp_tokens),
            'max': max(cpp_tokens),
            'mean': sum(cpp_tokens) / len(cpp_tokens),
        },
        'cuda_token_count': {
            'min': min(cuda_tokens),
            'max': max(cuda_tokens),
            'mean': sum(cuda_tokens) / len(cuda_tokens),
        },
    }


if __name__ == '__main__':
    # Test the dataset loader
    import argparse
    
    parser = argparse.ArgumentParser(description='Test C++ to CUDA dataset loader')
    parser.add_argument('--data_path', type=str, required=True, help='Path to JSONL file')
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    
    args = parser.parse_args()
    
    # Print statistics
    print("Dataset Statistics:")
    print("-" * 40)
    stats = get_dataset_statistics(args.data_path)
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\nLoading and splitting data:")
    print("-" * 40)
    train_data, test_data = load_cpp_cuda_data(
        args.data_path,
        train_ratio=args.train_ratio,
        test_ratio=args.test_ratio
    )
    
    print("\nSample data:")
    print("-" * 40)
    if train_data:
        sample = train_data[0]
        print(f"C++ code (first 200 chars): {sample['cpp'][:200]}...")
        print(f"CUDA code (first 200 chars): {sample['generated_cuda'][:200]}...")
