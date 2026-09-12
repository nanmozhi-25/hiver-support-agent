"""
Batch Reply Evaluation Module.
Generates replies for Golden Set messages and evaluates response quality using LLM Judge.
"""

import os
import json
import numpy as np
import pandas as pd

from src.pipeline import SupportAgentPipeline
from evaluation.llm_judge import LLMJudge

GOLDEN_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "golden", "golden_set.csv")


def evaluate_batch_replies(golden_csv: str = GOLDEN_CSV, max_samples: int = 200) -> dict:
    df_golden = pd.read_csv(golden_csv).head(max_samples)
    pipeline = SupportAgentPipeline()
    judge = LLMJudge()
    
    overall_scores = []
    correctness_scores = []
    grounding_scores = []
    relevance_scores = []
    actionability_scores = []
    tone_scores = []
    hallucination_scores = []
    
    for _, row in df_golden.iterrows():
        c_text = str(row["text"])
        res = pipeline.process(c_text)
        
        j_res = judge.evaluate_reply(
            customer_text=c_text,
            predicted_intent=res["intent"],
            retrieved_evidence=res["retrieved_evidence"],
            generated_reply=res["draft_reply"]
        )
        
        overall_scores.append(j_res["overall"])
        correctness_scores.append(j_res["correctness"])
        grounding_scores.append(j_res["grounding"])
        relevance_scores.append(j_res["relevance"])
        actionability_scores.append(j_res["actionability"])
        tone_scores.append(j_res["tone"])
        hallucination_scores.append(j_res["hallucination"])
        
    metrics = {
        "num_replies_evaluated": len(df_golden),
        "mean_overall_score": round(float(np.mean(overall_scores)), 2),
        "mean_correctness": round(float(np.mean(correctness_scores)), 2),
        "mean_grounding": round(float(np.mean(grounding_scores)), 2),
        "mean_relevance": round(float(np.mean(relevance_scores)), 2),
        "mean_actionability": round(float(np.mean(actionability_scores)), 2),
        "mean_tone": round(float(np.mean(tone_scores)), 2),
        "mean_hallucination_safety": round(float(np.mean(hallucination_scores)), 2)
    }
    
    print("Batch Reply Quality Evaluation (0-4 Scale):")
    print(f"  Overall Quality:       {metrics['mean_overall_score']:.2f} / 4.00")
    print(f"  Correctness:           {metrics['mean_correctness']:.2f} / 4.00")
    print(f"  Grounding Evidence:    {metrics['mean_grounding']:.2f} / 4.00")
    print(f"  Relevance:             {metrics['mean_relevance']:.2f} / 4.00")
    print(f"  Actionability:         {metrics['mean_actionability']:.2f} / 4.00")
    print(f"  Tone:                  {metrics['mean_tone']:.2f} / 4.00")
    print(f"  Hallucination Safety:  {metrics['mean_hallucination_safety']:.2f} / 4.00")
    
    return metrics


if __name__ == "__main__":
    evaluate_batch_replies()
