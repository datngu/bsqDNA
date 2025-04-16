import numpy as np
import glob
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def count_sequences_in_files(data_dir, num_files=5):
    """Count sequences in first N npz files"""
    files = sorted(glob.glob(str(Path(data_dir) / "*.npz")))[:num_files]
    logger.info(f"Examining {len(files)} files: {[Path(f).name for f in files]}")
    
    total_sequences = 0
    sequences_per_file = []
    
    for file_idx, file in enumerate(files):
        logger.info(f"Counting sequences in file: {file}")
        data = np.load(file)
        file_sequence_count = 0
        
        for key in data.keys():
            if data[key].dtype == np.dtype('|S1'):
                seq_count = len(data[key])
                file_sequence_count += seq_count
                logger.info(f"  Key {key}: {seq_count} sequences")
        
        total_sequences += file_sequence_count
        sequences_per_file.append(file_sequence_count)
        logger.info(f"  Total in file: {file_sequence_count} sequences")
    
    logger.info(f"Total sequences across {len(files)} files: {total_sequences}")
    logger.info(f"Sequences per file: {sequences_per_file}")
    
    # Calculate batches at different sizes
    batch_sizes = [32, 64, 128, 256, 512, 1024]
    logger.info("\nNumber of batches at different batch sizes:")
    for batch_size in batch_sizes:
        num_batches = total_sequences // batch_size
        remainder = total_sequences % batch_size
        logger.info(f"  Batch size {batch_size}: {num_batches} batches with {remainder} leftover sequences")
        
        # Check if this batch size would cause cache thrashing
        max_seqs_per_file = max(sequences_per_file)
        if batch_size > max_seqs_per_file:
            logger.warning(f"  ⚠️ Batch size {batch_size} exceeds max sequences in a single file ({max_seqs_per_file}).")
            logger.warning(f"    This could cause cache thrashing with a cache size of 5.")
    
    # Recommend batch size
    logger.info("\nRecommendation:")
    max_safe_batch_size = min(sequences_per_file)
    logger.info(f"For 5 cache entries, try batch size ≤ {max_safe_batch_size} to avoid cache thrashing")
    
    return total_sequences, sequences_per_file

if __name__ == "__main__":
    import sys
    
    # Get data directory from command line or use default
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    num_files = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    count_sequences_in_files(data_dir, num_files) 