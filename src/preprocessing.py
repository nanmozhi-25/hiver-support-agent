"""
Data Preprocessing & Data Leakage Prevention Module.
Implements conversation-level train/dev/test splitting, normalization, and sample generation.
"""

import os
import re
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict


def clean_text(text: str) -> str:
    """
    Cleans customer and brand tweet text while preserving support semantics.
    Removes Twitter handles (@user), extra whitespace, and standardizes text.
    """
    if not isinstance(text, str):
        return ""
    
    # Remove twitter handles (@mention)
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Replace multiple spaces/newlines
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def conversation_level_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    dev_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits data strictly at the conversation level to prevent data leakage.
    No messages from the same conversation will appear across splits.
    """
    assert abs(train_ratio + dev_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1.0"
    
    np.random.seed(random_seed)
    
    # Group by conversation_id
    if "conversation_id" in df.columns:
        unique_convs = df["conversation_id"].unique()
    else:
        # Fallback to customer_tweet_id if conversation_id missing
        unique_convs = df.index.values
        
    shuffled_convs = np.random.permutation(unique_convs)
    
    n_total = len(shuffled_convs)
    n_train = int(n_total * train_ratio)
    n_dev = int(n_total * dev_ratio)
    
    train_convs = set(shuffled_convs[:n_train])
    dev_convs = set(shuffled_convs[n_train:n_train + n_dev])
    test_convs = set(shuffled_convs[n_train + n_dev:])
    
    if "conversation_id" in df.columns:
        train_df = df[df["conversation_id"].isin(train_convs)].copy()
        dev_df = df[df["conversation_id"].isin(dev_convs)].copy()
        test_df = df[df["conversation_id"].isin(test_convs)].copy()
    else:
        train_df = df.loc[list(train_convs)].copy()
        dev_df = df.loc[list(dev_convs)].copy()
        test_df = df.loc[list(test_convs)].copy()
        
    print(f"Conversation-Level Split Completed:")
    print(f"  Train set: {len(train_df)} conversations ({len(train_convs)} unique IDs)")
    print(f"  Dev set:   {len(dev_df)} conversations ({len(dev_convs)} unique IDs)")
    print(f"  Test set:  {len(test_df)} conversations ({len(test_convs)} unique IDs)")
    
    return train_df, dev_df, test_df


def prepare_dataset(pairs_df: pd.DataFrame, output_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Cleans text, splits at conversation level, and saves split CSV files.
    """
    pairs_df["cleaned_customer_text"] = pairs_df["customer_text"].apply(clean_text)
    pairs_df["cleaned_brand_text"] = pairs_df["brand_text"].apply(clean_text)
    
    # Filter out empty cleaned customer texts
    valid_pairs = pairs_df[pairs_df["cleaned_customer_text"].str.len() >= 5].copy()
    
    train_df, dev_df, test_df = conversation_level_split(valid_pairs)
    
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    dev_df.to_csv(os.path.join(output_dir, "dev.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    
    return {
        "train": train_df,
        "dev": dev_df,
        "test": test_df
    }


if __name__ == "__main__":
    sample_text = "@AmazonHelp My order #12345 hasn't arrived yet! https://t.co/xyz"
    print("Sample cleaning result:")
    print(f"  Original: {sample_text}")
    print(f"  Cleaned:  '{clean_text(sample_text)}'")
