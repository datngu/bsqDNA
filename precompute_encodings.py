#!/usr/bin/env python
"""
Pre-compute one-hot encodings for DNA sequences.
This script converts all DNA sequence data from .npz files to pre-computed
one-hot encodings saved as PyTorch tensors (.pt files).

Uses GPU acceleration for faster encoding when available.
"""

import os
import glob
import time
import numpy as np
import torch
from pathlib import Path
import logging
from tqdm import tqdm
from datetime import datetime

from bsqdna.utils import GPUOneHotEncoder

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def precompute_encodings(data_dir: str="data", output_dir: str = "data_encoded", batch_size: int = 2048, 
                        max_chunk_size: int = 10000, device: str = None):
    """
    Pre-compute one-hot encodings for all DNA sequences in the data directory
    and save them as PyTorch tensors.
    
    Args:
        data_dir: Directory containing .npz files with DNA sequences
        output_dir: Directory to save the pre-computed tensors
        batch_size: Number of sequences to process at once
        max_chunk_size: Maximum number of sequences per output file
        device: Device to use for encoding ('cuda', 'cpu', etc.)
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Get all NPZ files
    files = sorted(glob.glob(str(data_dir / "*.npz")))
    logger.info(f"Found {len(files)} NPZ files in {data_dir}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    logger.info(f"Using device: {device}")
    encoder = GPUOneHotEncoder(device=device)
    
    # Process each file
    for file_idx, file_path in enumerate(files):
        file_name = Path(file_path).stem
        
        # Create a directory for this file's chunks
        file_output_dir = output_dir / file_name
        file_output_dir.mkdir(exist_ok=True, parents=True)
        
        # Check if we've already processed this file by looking for a "completed.txt" marker
        completed_marker = file_output_dir / "completed.txt"
        if completed_marker.exists():
            logger.info(f"Skipping {file_path} - already processed")
            continue
        
        logger.info(f"Processing file {file_idx+1}/{len(files)}: {file_path}")
        
        try:
            data = np.load(file_path)
            
            # Find DNA sequence keys
            dna_keys = []
            for key in data.keys():
                if data[key].dtype == np.dtype('|S1'):
                    dna_keys.append(key)
            
            if not dna_keys:
                logger.warning(f"No DNA sequences found in {file_path}, skipping")
                continue
            
            # Process each key
            for key in dna_keys:
                sequences = data[key]
                n_sequences = len(sequences)
                logger.info(f"  Found {n_sequences} sequences in key '{key}'")
                
                # Create a subdirectory for this key
                key_output_dir = file_output_dir / key
                key_output_dir.mkdir(exist_ok=True, parents=True)
                
                # Process in batches AND save in chunks
                batch_start_time = time.time()
                encoded_batch = []
                chunk_idx = 0
                processed_count = 0
                
                # Use tqdm for progress bar
                progress_bar = tqdm(range(0, n_sequences, batch_size), desc=f"Processing {key}")
                for batch_start in progress_bar:
                    batch_end = min(batch_start + batch_size, n_sequences)
                    batch = sequences[batch_start:batch_end]
                    current_batch_size = batch_end - batch_start
                    
                    # Process entire batch at once on GPU
                    if isinstance(batch, np.ndarray) and batch.ndim == 2:
                        # Process the whole batch at once
                        with torch.no_grad():
                            encoded_tensors = encoder(batch)
                            
                        # Add to current chunk 
                        for i in range(encoded_tensors.shape[0]):
                            encoded_batch.append(encoded_tensors[i])
                    else:
                        # Fall back to sequence-by-sequence processing
                        for seq in batch:
                            with torch.no_grad():
                                encoded_tensor = encoder.encode(seq)
                                encoded_batch.append(encoded_tensor)
                    
                    # If we've reached the max chunk size, save to disk
                    while len(encoded_batch) >= max_chunk_size:
                        chunk_tensors = encoded_batch[:max_chunk_size]
                        chunk_tensor = torch.stack(chunk_tensors)
                        chunk_file = key_output_dir / f"chunk_{chunk_idx:05d}.pt"
                        torch.save(chunk_tensor, chunk_file)
                        
                        logger.info(f"  Saved chunk {chunk_idx} with {len(chunk_tensors)} sequences to {chunk_file}")
                        encoded_batch = encoded_batch[max_chunk_size:]  # Keep remaining tensors
                        chunk_idx += 1
                    
                    processed_count += current_batch_size
                    progress_bar.set_postfix({"processed": processed_count, "chunks": chunk_idx})
                
                # Save any remaining encodings
                if encoded_batch:
                    chunk_tensor = torch.stack(encoded_batch)
                    chunk_file = key_output_dir / f"chunk_{chunk_idx:05d}.pt"
                    torch.save(chunk_tensor, chunk_file)
                    
                    logger.info(f"  Saved final chunk {chunk_idx} with {len(encoded_batch)} sequences to {chunk_file}")
                
                logger.info(f"  Total processing time for {key}: {time.time() - batch_start_time:.2f}s")
                
                # Create a metadata file with information about this key
                with open(key_output_dir / "metadata.txt", "w") as f:
                    f.write(f"original_file: {file_path}\n")
                    f.write(f"key: {key}\n")
                    f.write(f"total_sequences: {n_sequences}\n")
                    f.write(f"total_chunks: {chunk_idx + 1}\n")
                    f.write(f"max_chunk_size: {max_chunk_size}\n")
                    f.write(f"encoded_with_gpu: {device.type == 'cuda'}\n")
            
            # Mark this file as completed
            with open(completed_marker, "w") as f:
                f.write(f"Completed processing at {datetime.now().isoformat()}\n")
                f.write(f"Device used: {device}\n")
                
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            continue
    
    logger.info(f"All sequences processed and saved to {output_dir}")
    return output_dir

if __name__ == "__main__":
    from fire import Fire
    
    Fire(precompute_encodings) 