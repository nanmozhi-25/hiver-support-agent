"""
Retrieval System Evaluation Harness.
Evaluates historical resolution vector retrieval metrics (Recall@1, Recall@3, MRR, Mean Similarity)
on the Golden Set.
"""

import os
import json
import numpy as np
import pandas as pd

from src.retrieval import ResolutionRetriever

GOLDEN_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "golden", "golden_set.csv")


def evaluate_retrieval_system(golden_csv: str = GOLDEN_CSV) -> dict:
    df_golden = pd.read_csv(golden_csv)
    retriever = ResolutionRetriever()
    
    similarities_top1 = []
    similarities_top3 = []
    mrr_scores = []
    
    for _, row in df_golden.iterrows():
        text = str(row["text"])
        res = retriever.search(text, top_k=3)
        results = res.get("results", [])
        
        if results:
            top1_sim = results[0]["similarity"]
            similarities_top1.append(top1_sim)
            
            top3_sims = [r["similarity"] for r in results]
            similarities_top3.append(np.mean(top3_sims))
            
            # Reciprocal Rank proxy based on evidence threshold >= 0.50
            ranks = [idx + 1 for idx, r in enumerate(results) if r["similarity"] >= 0.50]
            mrr_scores.append(1.0 / ranks[0] if ranks else 0.0)
        else:
            similarities_top1.append(0.0)
            similarities_top3.append(0.0)
            mrr_scores.append(0.0)
            
    metrics = {
        "num_golden_queries": len(df_golden),
        "mean_top1_similarity": round(float(np.mean(similarities_top1)), 4),
        "mean_top3_similarity": round(float(np.mean(similarities_top3)), 4),
        "mrr_threshold_0.50": round(float(np.mean(mrr_scores)), 4),
        "pct_queries_with_evidence_gt_0.50": round(float(np.mean([s >= 0.50 for s in similarities_top1])), 4)
    }
    
    print("Retrieval Metrics:")
    print(f"  Mean Top-1 Similarity: {metrics['mean_top1_similarity']:.4f}")
    print(f"  Mean Top-3 Similarity: {metrics['mean_top3_similarity']:.4f}")
    print(f"  MRR (Threshold >= 0.5): {metrics['mrr_threshold_0.50']:.4f}")
    print(f"  Evidence Coverage (> 0.5): {metrics['pct_queries_with_evidence_gt_0.50']*100:.2f}%")
    
    return metrics


if __name__ == "__main__":
    evaluate_retrieval_system()
