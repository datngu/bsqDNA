from typing import Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
import glob
import os
from pathlib import Path
from .utils import DNAOneHotEncoder
import logging

logger = logging.getLogger(__name__)

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
        self.files = sorted(glob.glob(str(self.data_dir / "*.npz")))
        logger.info(f"Found {len(self.files)} .npz files in {data_dir}")
        
        self.encoder = DNAOneHotEncoder()
        
        # Load and concatenate all sequences
        sequences = []
        for file in self.files:
            logger.info(f"Loading file: {file}")
            data = np.load(file)
            for key in data.keys():
                if data[key].dtype == np.dtype('|S1'):
                    logger.info(f"Processing key {key} with shape {data[key].shape}")
                    sequences.append(data[key])
        
        if not sequences:
            raise ValueError(f"No DNA sequences found in {data_dir}")
            
        # Split the data into train/valid/test
        all_sequences = np.concatenate(sequences, axis=0)
        n_samples = len(all_sequences)
        logger.info(f"Total sequences loaded: {n_samples}")
        
        # Use fixed split ratios: 80% train, 10% valid, 10% test
        train_size = int(0.8 * n_samples)
        valid_size = int(0.1 * n_samples)
        
        if split == "train":
            self.sequences = all_sequences[:train_size]
            logger.info(f"Using {len(self.sequences)} sequences for training")
        elif split == "valid":
            self.sequences = all_sequences[train_size:train_size + valid_size]
            logger.info(f"Using {len(self.sequences)} sequences for validation")
        else:  # test
            self.sequences = all_sequences[train_size + valid_size:]
            logger.info(f"Using {len(self.sequences)} sequences for testing")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get sequence and convert to one-hot
        seq = self.sequences[idx]
        one_hot = self.encoder.encode(seq)
        labels = np.argmax(one_hot, axis=0)
        return torch.from_numpy(one_hot).float(), torch.from_numpy(labels).long()
