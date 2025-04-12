import numpy as np
import torch
import os
import glob
from pathlib import Path

# DNA one-hot encoding utility function
def one_hot_encode(sequences):
    """
    One-hot encode DNA sequences
    
    Args:
        sequences: numpy array of shape (batch_size, seq_length) with dtype='S1'
        
    Returns:
        one_hot: numpy array of shape (batch_size, 4, seq_length)
    """
    batch_size, seq_length = sequences.shape
    
    # Initialize one-hot encoding (4 possible nucleotides: A, C, G, T)
    one_hot = np.zeros((batch_size, 4, seq_length), dtype=np.float32)
    
    # Encode each nucleotide
    one_hot[:, 0, :] = (sequences == b'A')  # A
    one_hot[:, 1, :] = (sequences == b'C')  # C
    one_hot[:, 2, :] = (sequences == b'G')  # G
    one_hot[:, 3, :] = (sequences == b'T')  # T
    
    return one_hot

def load_and_check_test_data(data_dir='./test_data', file_pattern='*.npz', max_samples=5, convert_to_tensor=True):
    """
    Load test data, show its dimensions, and optionally convert to PyTorch tensor
    
    Args:
        data_dir: Directory containing test data
        file_pattern: Pattern to match data files
        max_samples: Maximum number of files to process
        convert_to_tensor: Whether to convert to PyTorch tensor
    """
    # Find all npz files in the data directory
    data_files = glob.glob(os.path.join(data_dir, file_pattern))
    print(f"Found {len(data_files)} data files matching pattern '{file_pattern}' in {data_dir}")
    
    # Process a subset of files
    for i, file_path in enumerate(data_files[:max_samples]):
        print(f"\n--- File {i+1}/{min(max_samples, len(data_files))}: {os.path.basename(file_path)} ---")
        
        # Load the data file
        data = np.load(file_path)
        print(f"Keys in file: {list(data.keys())}")
        
        # Process each key in the file
        for key in data.keys():
            array = data[key]
            print(f"\nArray: {key}")
            print(f"  Raw shape: {array.shape}")
            print(f"  Data type: {array.dtype}")
            
            if array.dtype == np.dtype('|S1'):  # DNA sequence data
                # Show a sample of the original data
                if array.shape[0] > 0:
                    sample_idx = 0
                    print(f"  Sample sequence (first 20 bases): {array[sample_idx, :20]}")
                
                # Convert to one-hot encoding
                print("  One-hot encoding...")
                one_hot = one_hot_encode(array[:min(10, array.shape[0])])  # Limit to 10 samples for memory
                print(f"  One-hot shape: {one_hot.shape}")
                
                # Convert to torch tensor if requested
                if convert_to_tensor:
                    print("  Converting to PyTorch tensor...")
                    tensor_data = torch.from_numpy(one_hot)
                    print(f"  Tensor shape: {tensor_data.shape}")
                    
                    # Add batch dimension for single samples if needed
                    if len(tensor_data.shape) == 2:
                        tensor_data = tensor_data.unsqueeze(0)
                        print(f"  Added batch dimension, new shape: {tensor_data.shape}")
                    
                    # For BSQ model input
                    print(f"  Final tensor dimension (B, C, L): {tensor_data.shape}")
                    
                    # For demonstration, you might want to see what your model expects
                    print("  This corresponds to:")
                    print(f"    Batch size: {tensor_data.shape[0]}")
                    print(f"    Channels (ACGT): {tensor_data.shape[1]}")
                    print(f"    Sequence length: {tensor_data.shape[2]}")
            else:
                print("  Not a DNA sequence array, skipping one-hot encoding.")

if __name__ == "__main__":
    load_and_check_test_data() 