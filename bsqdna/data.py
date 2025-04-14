import numpy as np
import torch
from torch.utils.data import Dataset
import glob
import os
from pathlib import Path
from .utils import DNAOneHotEncoder

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
        self.data_dir = Path(data_dir)
        self.files = sorted(glob.glob(str(self.data_dir / "*.npz")))
        self.encoder = DNAOneHotEncoder()
        
        # Load and concatenate all sequences
        sequences = []
        for file in self.files:
            data = np.load(file)
            for key in data.keys():
                if data[key].dtype == np.dtype('|S1'):
                    sequences.append(data[key])
        
        # Split the data into train/valid/test
        all_sequences = np.concatenate(sequences, axis=0)
        n_samples = len(all_sequences)
        
        # Use fixed split ratios: 80% train, 10% valid, 10% test
        train_size = int(0.8 * n_samples)
        valid_size = int(0.1 * n_samples)
        
        if split == "train":
            self.sequences = all_sequences[:train_size]
        elif split == "valid":
            self.sequences = all_sequences[train_size:train_size + valid_size]
        else:  # test
            self.sequences = all_sequences[train_size + valid_size:]
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        # Get sequence and convert to one-hot
        seq = self.sequences[idx]
        one_hot = self.encoder.encode(seq)
        return torch.from_numpy(one_hot).float()
