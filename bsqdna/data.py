from typing import Tuple, List, Dict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import glob
import os
from pathlib import Path
from .utils import DNAOneHotEncoder
import logging
import re
import time

logger = logging.getLogger(__name__)

def dna_collate_fn(batch):
    """
    Custom collate function for DNA sequences to ensure proper batching
    
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

def create_dataloader(data_dir: str, split: str = "train", batch_size: int = 32, 
                     num_workers: int = 4, shuffle: bool = True) -> DataLoader:
    """
    Create a DataLoader for DNA sequences with the correct collate function
    
    Args:
        data_dir: Root directory containing .npz files
        split: One of 'train', 'valid', or 'test'
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        shuffle: Whether to shuffle the data
        
    Returns:
        DataLoader: PyTorch DataLoader with custom collate function
    """
    dataset = DNADataset(data_dir, split)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=dna_collate_fn,
        pin_memory=False
    )

class DNADataset(Dataset):
    """
    Dataset for loading DNA sequences from .npz files
    """
    def __init__(self, data_dir: str, split: str = "train"):
        """
        Args:
            data_dir: Root directory containing .npz files
            split: One of 'train', 'valid', or 'test'
        """
        logger.info(f"Initializing DNADataset with data_dir={data_dir}, split={split}")
        self.data_dir = Path(data_dir)
        self.files = sorted(glob.glob(str(self.data_dir / "*.npz")), 
                           key=lambda f: [int(s) if s.isdigit() else s.lower() 
                                        for s in re.split(r'(\d+)', os.path.basename(f))])
        logger.info(f"Found {len(self.files)} .npz files in {data_dir}")
        
        self.encoder = DNAOneHotEncoder()
        
        self.file_index: List[Dict] = []
        total_sequences = 0
        
        for file_idx, file in enumerate(self.files):
            logger.info(f"Indexing file: {file}")
            data = np.load(file)
            for key in data.keys():
                if data[key].dtype == np.dtype('|S1'):
                    seq_count = len(data[key])
                    total_sequences += seq_count
                    self.file_index.append({
                        'file_path': file,
                        'key': key,
                        'count': seq_count,
                        'start_idx': total_sequences - seq_count,
                        'end_idx': total_sequences - 1
                    })
        
        if not self.file_index:
            raise ValueError(f"No DNA sequences found in {data_dir}")
            
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
    
    def _load_file(self, file_path, key):
        """Load a file, with caching"""
        cache_key = f"{file_path}:{key}"
        if cache_key not in self.cache:
            # If cache is full, remove the least recently used item
            if len(self.cache) >= self.max_cache_size:
                # Remove first item (least recently used)
                self.cache.pop(next(iter(self.cache)))
            
            # Load the file
            data = np.load(file_path)
            self.cache[cache_key] = data[key]
            
        return self.cache[cache_key]
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        start_time = time.time()
        
        if idx >= self.length:
            raise IndexError(f"Index {idx} out of range for dataset of length {self.length}")
        
        # Find which file contains this sequence
        file_info, seq_idx = self._find_file_info(idx)
        
        file_find_time = time.time() - start_time
        start_time = time.time()
        
        # Load the sequence
        sequences = self._load_file(file_info['file_path'], file_info['key'])
        seq = sequences[seq_idx]
        
        file_load_time = time.time() - start_time
        start_time = time.time()
        
        # Convert to one-hot
        # Ensure seq is a numpy array, not a bytes object
        if isinstance(seq, bytes):
            seq = np.array([s for s in seq], dtype='S1')
            
        one_hot = self.encoder.encode(seq)
        
        # Make sure one_hot is a proper numpy array
        if isinstance(one_hot, list):
            one_hot = np.array(one_hot)
            
        labels = np.argmax(one_hot, axis=0)
        
        # Convert to tensors - ensure correct dimensions (C, L)
        one_hot_tensor = torch.from_numpy(one_hot).float()
        labels_tensor = torch.from_numpy(labels).long()
        
        encoding_time = time.time() - start_time
        
        # Print times occasionally
        if idx % 1000 == 0:
            print(f"Item {idx}: Find={file_find_time:.4f}s, Load={file_load_time:.4f}s, Encode={encoding_time:.4f}s")
        
        return one_hot_tensor, labels_tensor
