"""
Master Evaluation Suite Orchestrator.
Executes complete evaluation suite across intent classification, vector retrieval, escalation engine,
reply generation quality, LLM judge, and human agreement. Outputs unified metrics to results/metrics.json.
"""

import os
import json
import argparse
import time

from baselines.majority_baseline import run_majority_baseline
from baselines.tfidf_logistic import train_and_eval_tfidf_logistic
from evaluation.evaluate_intent import evaluate_intent_classifier
from evaluation.evaluate_retrieval import evaluate_retrieval_system
from evaluation.evaluate_escalation import evaluate_escalation_engine
from evaluation.evaluate_replies import evaluate_batch_replies
from evaluation.human_agreement import evaluate_human_llm_agreement

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
METRICS_JSON_PATH = os.path.join(RESULTS_DIR, "metrics.json")


def run_full_evaluation_suite(quick_mode: bool = False) -> dict:
    start_time = time.time()
    print("=========================================================================")
    print("      HIVER AI SUPPORT AGENT — MASTER EVALUATION SUITE RUNNER           ")
    print("=========================================================================\n")
    
    # 1. Trivial Majority Baseline
    print("--- [1/6] Running Majority Class Baseline ---")
    maj_metrics = run_majority_baseline()
    print()
    
    # 2. Simple ML Baseline (TF-IDF + Logistic Regression)
    print("--- [2/6] Running TF-IDF + Logistic Regression Baseline ---")
    tfidf_metrics, _, _ = train_and_eval_tfidf_logistic()
    print()
    
    # 3. Main Intent Classifier Evaluation
    print("--- [3/6] Running Main Intent Classifier Evaluation ---")
    main_intent_metrics = evaluate_intent_classifier()
    print()
    
    # 4. Historical Resolution Retrieval Evaluation
    print("--- [4/6] Running Vector Retrieval Evaluation ---")
    retrieval_metrics = evaluate_retrieval_system()
    print()
    
    # 5. Escalation Decision Engine Evaluation
    print("--- [5/6] Running Escalation Engine Evaluation ---")
    escalation_metrics = evaluate_escalation_engine()
    print()
    
    # 6. Reply Quality & Human Agreement Evaluation
    print("--- [6/6] Running Reply Quality & Human Agreement Evaluation ---")
    max_reply_samples = 50 if quick_mode else 200
    reply_metrics = evaluate_batch_replies(max_samples=max_reply_samples)
    human_metrics = evaluate_human_llm_agreement()
    print()
    
    elapsed = time.time() - start_time
    
    # Consolidate all empirical metrics
    full_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_execution_seconds": round(elapsed, 2),
        "baselines": {
            "majority_baseline": maj_metrics,
            "tfidf_logistic_baseline": tfidf_metrics
        },
        "main_system": {
            "intent_classifier": main_intent_metrics,
            "retrieval": retrieval_metrics,
            "escalation_engine": escalation_metrics,
            "reply_quality": reply_metrics,
            "human_llm_agreement": human_metrics
        }
    }
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(METRICS_JSON_PATH, "w") as f:
        json.dump(full_results, f, indent=2)
        
    print("=========================================================================")
    print(f"EVALUATION SUITE COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS!")
    print(f"All empirical metrics saved to: {METRICS_JSON_PATH}")
    print("=========================================================================")
    
    return full_results


def main():
    parser = argparse.ArgumentParser(description="Run complete evaluation harness for Hiver AI Support Agent.")
    parser.add_argument("--quick", action="store_true", help="Run fast evaluation mode (< 2 mins)")
    args = parser.parse_args()
    
    run_full_evaluation_suite(quick_mode=args.quick)


if __name__ == "__main__":
    main()
