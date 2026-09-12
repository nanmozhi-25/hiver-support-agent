"""
Data Preparation CLI Script.
Reconstructs AmazonHelp conversations, performs conversation-level splitting,
and saves dataset splits to data/processed/.
"""

import os
import argparse
import pandas as pd
from src.data_loader import inspect_and_rank_brands, reconstruct_conversations, RAW_DATA_PATH
from src.preprocessing import prepare_dataset

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def main(brand: str = "AmazonHelp", sample_size: int = 5000):
    print(f"=== Phase 1 & 2: Data Preparation for Brand '{brand}' ===")
    
    # 1. Run / verify brand analysis
    analysis_csv = os.path.join(os.path.dirname(__file__), "..", "data", "brand_analysis.csv")
    if not os.path.exists(analysis_csv):
        inspect_and_rank_brands(csv_path=RAW_DATA_PATH, output_csv=analysis_csv)
        
    # 2. Reconstruct conversations
    pairs_df = reconstruct_conversations(brand=brand, csv_path=RAW_DATA_PATH, max_conversations=sample_size)
    
    if len(pairs_df) == 0:
        raise ValueError(f"No conversation pairs found for brand {brand}!")
        
    # Save full reconstructed pairs
    pairs_csv = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "pairs.csv")
    os.makedirs(os.path.dirname(pairs_csv), exist_ok=True)
    pairs_df.to_csv(pairs_csv, index=False)
    print(f"Saved {len(pairs_df)} raw pairs to {pairs_csv}")
    
    # 3. Preprocess and split at conversation level
    splits = prepare_dataset(pairs_df, PROCESSED_DIR)
    
    print("\nData Preparation Finished Successfully:")
    print(f"  Train samples: {len(splits['train'])}")
    print(f"  Dev samples:   {len(splits['dev'])}")
    print(f"  Test samples:  {len(splits['test'])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset for selected brand.")
    parser.add_argument("--brand", type=str, default="AmazonHelp", help="Selected brand handle")
    parser.add_argument("--sample-size", type=int, default=5000, help="Number of conversation pairs to extract")
    args = parser.parse_args()
    
    main(brand=args.brand, sample_size=args.sample_size)
