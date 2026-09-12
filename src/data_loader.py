"""
Dataset Loader & Brand Analyzer for Twitter Customer Support Dataset (twcs.csv).
Analyzes brand distributions, conversation counts, text noise, and extracts structured multi-turn conversations.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

RAW_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "twcs.csv")
BRAND_ANALYSIS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "brand_analysis.csv")


def inspect_and_rank_brands(csv_path: str = RAW_DATA_PATH, output_csv: str = BRAND_ANALYSIS_PATH) -> pd.DataFrame:
    """
    Inspects twcs.csv to calculate brand statistics, conversation counts, noise metrics,
    and ranks candidate brands.
    """
    print(f"Loading raw dataset from {csv_path}...")
    df = pd.read_csv(csv_path, dtype={"tweet_id": str, "in_response_to_tweet_id": str, "response_tweet_id": str})
    
    print(f"Total dataset size: {len(df):,} tweets.")
    
    # Identify brand handles (inbound == False -> author is a brand)
    brand_df = df[df["inbound"] == False]
    brand_counts = brand_df["author_id"].value_counts()
    
    analysis_records = []
    
    print("Analyzing top 30 brands...")
    top_brands = brand_counts.head(30).index.tolist()
    
    for brand in top_brands:
        brand_mask = (df["author_id"] == brand)
        b_tweets = df[brand_mask]
        
        # Find customer tweets responding to this brand or responded to by this brand
        b_tweet_ids = set(b_tweets["tweet_id"].dropna())
        
        # Inbound tweets replying to brand's tweets or brand replying to inbound tweets
        customer_replies = df[(df["inbound"] == True) & (df["in_response_to_tweet_id"].isin(b_tweet_ids))]
        
        total_brand_tweets = len(b_tweets)
        total_customer_tweets = len(customer_replies)
        total_messages = total_brand_tweets + total_customer_tweets
        
        # Combine texts for noise metrics
        sample_texts = pd.concat([b_tweets["text"], customer_replies["text"]]).dropna()
        
        missing_count = b_tweets["text"].isna().sum() + customer_replies["text"].isna().sum()
        dup_count = sample_texts.duplicated().sum()
        dup_pct = round((dup_count / max(len(sample_texts), 1)) * 100, 2)
        
        word_counts = sample_texts.apply(lambda x: len(str(x).split()))
        avg_words = round(word_counts.mean(), 2) if len(word_counts) > 0 else 0
        short_msg_pct = round((word_counts < 4).mean() * 100, 2) if len(word_counts) > 0 else 0
        
        # Estimate multi-turn conversations (inbound tweets that initiated or replied)
        num_conversations = total_customer_tweets
        
        # Rank score balancing volume, average length, and low duplicate noise
        score = num_conversations * (1.0 - (dup_pct / 100.0)) * (1.0 - (short_msg_pct / 100.0))
        
        analysis_records.append({
            "brand": brand,
            "total_tweets": total_messages,
            "brand_tweets": total_brand_tweets,
            "customer_tweets": total_customer_tweets,
            "est_conversations": num_conversations,
            "avg_word_count": avg_words,
            "short_msg_pct": short_msg_pct,
            "duplicate_pct": dup_pct,
            "missing_text_count": missing_count,
            "rank_score": round(score, 2)
        })
        
    analysis_df = pd.DataFrame(analysis_records)
    analysis_df = analysis_df.sort_values(by="rank_score", ascending=False).reset_index(drop=True)
    analysis_df["rank"] = analysis_df.index + 1
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    analysis_df.to_csv(output_csv, index=False)
    print(f"Brand analysis saved to {output_csv}")
    
    return analysis_df


def reconstruct_conversations(brand: str, csv_path: str = RAW_DATA_PATH, max_conversations: Optional[int] = 5000) -> pd.DataFrame:
    """
    Extracts customer-support pairs (Customer Query -> Brand Response) for a specific brand.
    Reconstructs full multi-turn context where available.
    """
    df = pd.read_csv(csv_path, dtype={"tweet_id": str, "in_response_to_tweet_id": str, "response_tweet_id": str})
    
    # Filter brand responses
    brand_responses = df[(df["inbound"] == False) & (df["author_id"] == brand)].copy()
    
    # Map tweet_id -> row for quick lookup
    tweet_dict = df.set_index("tweet_id").to_dict(orient="index")
    
    pairs = []
    seen_conversations = set()
    
    for _, b_row in brand_responses.iterrows():
        b_id = b_row["tweet_id"]
        in_resp_id = b_row["in_response_to_tweet_id"]
        
        if pd.isna(in_resp_id) or in_resp_id not in tweet_dict:
            continue
            
        c_row = tweet_dict[in_resp_id]
        if not c_row["inbound"]:  # Ensure reply target is a customer message
            continue
            
        c_id = in_resp_id
        conv_id = f"{c_id}_{b_id}"
        
        if conv_id in seen_conversations:
            continue
        seen_conversations.add(conv_id)
        
        c_text = str(c_row["text"]).strip()
        b_text = str(b_row["text"]).strip()
        
        # Basic validation: discard empty or non-informative
        if len(c_text) < 5 or len(b_text) < 5:
            continue
            
        pairs.append({
            "conversation_id": conv_id,
            "brand": brand,
            "customer_tweet_id": c_id,
            "brand_tweet_id": b_id,
            "customer_text": c_text,
            "brand_text": b_text,
            "customer_created_at": c_row.get("created_at", ""),
            "brand_created_at": b_row.get("created_at", "")
        })
        
        if max_conversations and len(pairs) >= max_conversations:
            break
            
    pairs_df = pd.DataFrame(pairs)
    print(f"Reconstructed {len(pairs_df)} customer-brand conversation pairs for {brand}.")
    return pairs_df


if __name__ == "__main__":
    df_analysis = inspect_and_rank_brands()
    print("\nTop 5 Candidate Brands:")
    print(df_analysis[["rank", "brand", "customer_tweets", "est_conversations", "avg_word_count", "rank_score"]].head())
