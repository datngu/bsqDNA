from typing import Tuple, List, Dict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import glob
import os
from pathlib import Path
import logging
import re

logger = logging.getLogger(__name__)

def encoded_collate_fn(batch):
    """
    Custom collate function for pre-encoded DNA sequences
    
    Args:
        batch: List of (one_hot, labels) tuples from __getitem__
        
    Returns:
        Tuple of (batched_one_hot, batched_labels) tensors
    """
    one_hot_tensors, label_tensors = zip(*batch)
    
    # Stack tensors along batch dimension
    batched_one_hot = torch.stack(one_hot_tensors)
    batched_labels = torch.stack(label_tensors)
    
    return batched_one_hot, batched_labels

def create_encoded_dataloader(data_dir: str, split: str = "train", batch_size: int = 32, 
                            num_workers: int = 16, shuffle: bool = True) -> DataLoader:
    """
    Create a DataLoader for pre-encoded DNA sequences
    
    Args:
        data_dir: Directory containing pre-computed .pt tensor files
        split: One of 'train', 'valid', or 'test'
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        shuffle: Whether to shuffle the data
        
    Returns:
        DataLoader: PyTorch DataLoader with custom collate function
    """
    dataset = EncodedDNADataset(data_dir, split)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=encoded_collate_fn,
        pin_memory=False,
        prefetch_factor=2
    )

class EncodedDNADataset(Dataset):
    """
    Dataset for loading pre-computed one-hot encoded DNA sequences
    """
    def __init__(self, data_dir: str, split: str = "train"):
        """
        Args:
            data_dir: Directory containing pre-computed .pt tensor files
            split: One of 'train', 'valid', or 'test'
        """
        logger.info(f"Initializing EncodedDNADataset with data_dir={data_dir}, split={split}")
        self.data_dir = Path(data_dir)
        
        # Look for chunked data structure first (directories with chunk files)
        self.chunked_format = False
        self.data_dirs = sorted([d for d in self.data_dir.glob("*") if d.is_dir()])
        
        if self.data_dirs:
            logger.info(f"Found {len(self.data_dirs)} data directories in {data_dir}, using chunked format")
            self.chunked_format = True
        else:
            # Fall back to the original format (flat .pt files)
            self.files = sorted(glob.glob(str(self.data_dir / "*.pt")))
            logger.info(f"Found {len(self.files)} .pt files in {data_dir}, using flat format")
            
            if not self.files:
                raise ValueError(f"No pre-computed tensor files found in {data_dir}")
        
        # Create index of all sequences
        self.file_index: List[Dict] = []
        total_sequences = 0
        
        if self.chunked_format:
            # Process chunk directories
            for dir_idx, dir_path in enumerate(self.data_dirs):
                logger.info(f"Indexing directory: {dir_path}")
                
                # Find all subdirectories (keys)
                key_dirs = sorted([d for d in dir_path.glob("*") if d.is_dir()])
                
                for key_dir in key_dirs:
                    # Find all chunk files in this key directory
                    chunk_files = sorted(glob.glob(str(key_dir / "chunk_*.pt")))
                    
                    for chunk_idx, chunk_file in enumerate(chunk_files):
                        logger.info(f"  Indexing chunk file: {chunk_file}")
                        
                        # Load chunk size information without loading the whole tensor
                        tensor_shape = torch.load(chunk_file, map_location=torch.device('cpu')).shape
                        chunk_size = tensor_shape[0]  # First dimension is batch size
                        
                        total_sequences += chunk_size
                        self.file_index.append({
                            'file_path': chunk_file,
                            'count': chunk_size,
                            'start_idx': total_sequences - chunk_size,
                            'end_idx': total_sequences - 1
                        })
        else:
            # Original flat file format
            for file_idx, file_path in enumerate(self.files):
                logger.info(f"Indexing file: {file_path}")
                
                # Load file metadata without loading entire tensor
                tensor_shape = torch.load(file_path, map_location=torch.device('cpu')).shape
                seq_count = tensor_shape[0]  # First dimension is batch size
                
                total_sequences += seq_count
                self.file_index.append({
                    'file_path': file_path,
                    'count': seq_count,
                    'start_idx': total_sequences - seq_count,
                    'end_idx': total_sequences - 1
                })
        
        # Calculate split indices based on total count
        logger.info(f"Total sequences indexed: {total_sequences}")
        
        # Use fixed split ratios: 80% train, 10% valid, 10% test
        train_size = int(0.8 * total_sequences)
        valid_size = int(0.1 * total_sequences)
        
        if split == "train":
            self.start_idx = 0
            self.end_idx = train_size - 1
            logger.info(f"Using sequences 0-{self.end_idx} for training")
        elif split == "valid":
            self.start_idx = train_size
            self.end_idx = train_size + valid_size - 1
            logger.info(f"Using sequences {self.start_idx}-{self.end_idx} for validation")
        else:  # test
            self.start_idx = train_size + valid_size
            self.end_idx = total_sequences - 1
            logger.info(f"Using sequences {self.start_idx}-{self.end_idx} for testing")
            
        self.length = self.end_idx - self.start_idx + 1
        
        # Cache for frequently accessed files
        self.cache = {}
        self.max_cache_size = 5
    
    def __len__(self):
        return self.length
    
    def _find_file_info(self, idx):
        """Find which file contains the sequence at the given index"""
        real_idx = idx + self.start_idx
        for file_info in self.file_index:
            if file_info['start_idx'] <= real_idx <= file_info['end_idx']:
                return file_info, real_idx - file_info['start_idx']
        raise IndexError(f"Index {idx} out of bounds")
    
    def _load_file(self, file_path):
        """Load a tensor file, with caching"""
        if file_path not in self.cache:
            # If cache is full, remove the least recently used item
            if len(self.cache) >= self.max_cache_size:
                # Remove first item (least recently used)
                self.cache.pop(next(iter(self.cache)))
            
            # Load the file
            self.cache[file_path] = torch.load(file_path)
            
        return self.cache[file_path]
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx >= self.length:
            raise IndexError(f"Index {idx} out of range for dataset of length {self.length}")
        
        # Find which file contains this sequence
        file_info, seq_idx = self._find_file_info(idx)
        
        # Load the tensors
        tensors = self._load_file(file_info['file_path'])
        one_hot_tensor = tensors[seq_idx]
        
        # Generate labels from one-hot encoding
        labels_tensor = torch.argmax(one_hot_tensor, dim=0)
        
        return one_hot_tensor, labels_tensor 