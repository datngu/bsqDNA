import argparse
import pysam
import numpy as np

def fasta_to_np(genome_path, chroms):
    fasta = pysam.FastaFile(genome_path)
    data = {}
    for chrom in chroms:
        seq = fasta.fetch(chrom)  # Fetch chromosome sequence
        seq_list = list(seq)  # Convert to list of characters
        data[chrom] = np.array(seq_list, dtype="S1")  # Convert to NumPy array with dtype 'S1'
    np.savez_compressed(output_path, **data)  # Save as npz file
    fasta.close()
    return data


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Convert genome FASTA to NumPy arrays.")
    parser.add_argument("--genome", required=True, help="Path to the genome FASTA file.")
    parser.add_argument("--out", required=True, help="Out .npz file.")
    parser.add_argument("--chroms", nargs='+', required=True, help="Space-separated list of chromosomes (e.g., --chroms chr1 chr2 chr3)")
    args = parser.parse_args()
    data = fasta_to_npz(args.genome, args.chroms, args.out)

    
