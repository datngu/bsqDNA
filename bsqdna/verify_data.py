#!/usr/bin/env python
"""
Utility script to verify integrity of DNA sequence data files
"""
import argparse
import json
import logging
import sys
from pathlib import Path

from .data import DNADataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Verify integrity of DNA sequence data files")
    parser.add_argument("data_dir", type=str, help="Directory containing .npz data files")
    parser.add_argument("--output", "-o", type=str, help="Output file for detailed report (JSON)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info(f"Verifying data integrity in directory: {args.data_dir}")
    
    try:
        results = DNADataset.verify_data_integrity(args.data_dir)
        
        # Print summary
        print("\n=== Data Integrity Report ===")
        print(f"Directory: {args.data_dir}")
        print(f"Total files: {results['total_files']}")
        print(f"Valid files: {results['valid_files']}")
        print(f"Corrupted files: {len(results['corrupted_files'])}")
        print(f"Total valid sequences: {results['valid_sequences']}")
        
        if results['corrupted_files']:
            print("\nCorrupted files:")
            for file in results['corrupted_files']:
                print(f"  - {file}")
        
        # Save detailed report if requested
        if args.output:
            output_path = Path(args.output)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Detailed report saved to {output_path}")
        
        if len(results['corrupted_files']) > 0:
            return 1
        return 0
        
    except Exception as e:
        logger.error(f"Error during verification: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 