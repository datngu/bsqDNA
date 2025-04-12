import argparse
import pysam
import numpy as np
import os
from pathlib import Path


def fasta_to_np(genome_path, chroms):
    fasta = pysam.FastaFile(genome_path)
    data = {}
    for chrom in chroms:
        seq = fasta.fetch(chrom)  # Fetch chromosome sequence
        seq_list = list(seq)  # Convert to list of characters
        data[chrom] = np.array(seq_list, dtype="S1")  # Convert to NumPy array with dtype 'S1'
    fasta.close()
    return data

def bin_genome_and_save(seq_data, chrom_name, output_dir, bin_size=4096, step=512):
    #seq_data = data[chrom_name]
    os.makedirs(output_dir, exist_ok=True)
    chrom_size = len(seq_data)
    bins = np.arange(0, chrom_size - bin_size + 1, step)
    
    seqs = []
    for start in bins:
        end = start + bin_size
        seq = seq_data[start:end]
        if b'N' not in seq:
            seqs.append(seq)

    seqs = np.stack(seqs)
    out_fn = Path(output_dir) / f'{chrom_name}.npz'
    np.savez_compressed(out_fn, seqs=seqs)
    print(f"Saved {chrom_name} with #seq = {len(seqs)} to {out_fn}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert genome FASTA to NumPy arrays and bin into 512bp chunks.")
    parser.add_argument("--genome", required=True, help="Path to the genome FASTA file.")
    parser.add_argument("--out_dir", required=True, help="Output directory for .npz files.")
    parser.add_argument("--chroms", nargs='+', required=True, help="Space-separated list of chromosomes (e.g., --chroms chr1 chr2 chr3)")
    args = parser.parse_args()
    
    data = fasta_to_np(args.genome, args.chroms)
    
    for chrom, seq_data in data.items():
        bin_genome_and_save(seq_data, chrom, args.out_dir)
