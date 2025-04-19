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
import h5py

logger = logging.getLogger(__name__)

def collate_fn(batch):
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
    Create a DataLoader for DNA sequences with optimized I/O handling
    
    Args:
        data_dir: Root directory containing .h5 files
        split: One of 'train', 'valid', or 'test'
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading. 
                    Set to 0 for single-process loading.
                    Recommended: 4 * number of GPUs
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
        collate_fn=collate_fn,
        pin_memory=True,  # Faster GPU transfer
        persistent_workers=True,  # Keep workers alive between epochs
        prefetch_factor=2  # Number of batches to prefetch per worker
    )

class DNADataset(Dataset):
    """
    Dataset for loading DNA sequences from HDF5 files with memory mapping
    """
    def __init__(self, data_dir: str, split: str = "train"):
        """
        Args:
            data_dir: Root directory containing .h5 files
            split: One of 'train', 'valid', or 'test'
        """
        self.data_dir = Path(data_dir)
        self.files = sorted(glob.glob(str(self.data_dir / "*.h5")), 
                           key=lambda f: [int(s) if s.isdigit() else s.lower() 
                                        for s in re.split(r'(\d+)', os.path.basename(f))])
        self.encoder = DNAOneHotEncoder()
        self.h5_files = {}
        self.file_index = []

        # Open all HDF5 files
        for file in self.files:
            self.h5_files[file] = h5py.File(file, 'r')
        
        # Index all sequences for split: 80% train, 10% validate, 10% test
        total_sequences = 0
        for file_idx, file in enumerate(self.files):
            h5f = self.h5_files[file]
            for key in h5f.keys():
                if h5f[key].dtype == np.dtype('|S1'):
                    seq_count = len(h5f[key])
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

        train_size = int(0.8 * total_sequences)
        valid_size = int(0.1 * total_sequences)
        
        if split == "train":
            self.start_idx = 0
            self.end_idx = train_size - 1
            logger.info(f"Using sequences 0-{self.end_idx} for training")
        elif split == "val":
            self.start_idx = train_size
            self.end_idx = train_size + valid_size - 1
            logger.info(f"Using sequences {self.start_idx}-{self.end_idx} for validation")
        else:
            self.start_idx = train_size + valid_size
            self.end_idx = total_sequences - 1
            logger.info(f"Using sequences {self.start_idx}-{self.end_idx} for testing")
            
        self.length = self.end_idx - self.start_idx + 1
    
    def __len__(self):
        return self.length
    
    def _find_file_info(self, idx):
        """Find which file contains the sequence at the given index"""
        real_idx = idx + self.start_idx
        for file_info in self.file_index:
            if file_info['start_idx'] <= real_idx <= file_info['end_idx']:
                return file_info, real_idx - file_info['start_idx']
        raise IndexError(f"Index {idx} out of bounds")
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx >= self.length:
            raise IndexError(f"Index {idx} out of range for dataset of length {self.length}")
        
        # Find which file contains this sequence
        file_info, seq_idx = self._find_file_info(idx)
        
        # Load the sequence using memory mapping
        h5f = self.h5_files[file_info['file_path']]
        seq = h5f[file_info['key']][seq_idx]
        
        # Convert to one-hot
        one_hot = self.encoder.encode(seq)
        labels = np.argmax(one_hot, axis=0)
        
        # Convert to tensors
        one_hot_tensor = torch.from_numpy(one_hot).float()
        labels_tensor = torch.from_numpy(labels).long()
        
        return one_hot_tensor, labels_tensor
    
    def __del__(self):
        """Close all HDF5 files when the dataset is deleted"""
        for h5f in self.h5_files.values():
            h5f.close()
