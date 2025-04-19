import numpy as np
import h5py
import glob
import os
from pathlib import Path
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

def convert_npz_to_h5(npz_dir: str="data", h5_dir: str="data_h5", chunk_size: int = 5000):
    """
    Convert all NPZ files in a directory to HDF5 format.
    
    Args:
        npz_dir: Directory containing .npz files
        h5_dir: Directory to save .h5 files
        chunk_size: Number of sequences to process at once
    """
    npz_dir = Path(npz_dir)
    h5_dir = Path(h5_dir)
    h5_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all NPZ files
    npz_files = sorted(glob.glob(str(npz_dir / "*.npz")))
    logger.info(f"Found {len(npz_files)} NPZ files to convert")
    
    for npz_file in tqdm(npz_files, desc="Converting files"):
        # Create corresponding HDF5 filename
        h5_file = h5_dir / (Path(npz_file).stem + ".h5")
        
        # Load NPZ data
        with np.load(npz_file) as data:
            # Create HDF5 file
            with h5py.File(h5_file, 'w') as h5f:
                for key in data.keys():
                    shape = data[key].shape
                    dtype = data[key].dtype

                    if dtype == np.dtype('|S1'):
                        # Create a resizable dataset
                        maxshape = (None,) + shape[1:] if len(shape) > 1 else (None,)
                        dataset = h5f.create_dataset(
                            key,
                            shape=(0,) + shape[1:] if len(shape) > 1 else (0,),
                            maxshape=maxshape,
                            dtype=dtype,
                            compression='gzip',
                            compression_opts=9
                        )
                        
                        # Process in chunks
                        total_sequences = shape[0]
                        for i in tqdm(range(0, total_sequences, chunk_size), 
                                    desc=f"Processing {key}", 
                                    leave=False):
                            end_idx = min(i + chunk_size, total_sequences)
                            chunk = data[key][i:end_idx]
                            
                            # Resize dataset and write chunk
                            dataset.resize(end_idx, axis=0)
                            dataset[i:end_idx] = chunk
                            
                            # Force flush to disk
                            h5f.flush()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from fire import Fire
    Fire(convert_npz_to_h5)