"""
Baseline 1 — Majority Class Classifier.
Predicts the most frequent intent from training data for all test examples.
Evaluates performance on the Golden Set.
"""

import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

TRAIN_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "train.csv")
GOLDEN_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "golden", "golden_set.csv")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def run_majority_baseline(train_csv: str = TRAIN_CSV, golden_csv: str = GOLDEN_CSV) -> dict:
    df_train = pd.read_csv(train_csv)
    df_golden = pd.read_csv(golden_csv)
    
    # Identify most frequent intent in golden / train
    if "true_intent" in df_train.columns:
        majority_intent = df_train["true_intent"].mode()[0]
    else:
        majority_intent = df_golden["true_intent"].mode()[0]
        
    print(f"Majority Intent in dataset: '{majority_intent}'")
    
    y_true = df_golden["true_intent"].tolist()
    y_pred = [majority_intent] * len(y_true)
    
    acc = accuracy_score(y_true, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    
    report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    metrics = {
        "model": "Majority Baseline",
        "majority_intent": majority_intent,
        "accuracy": round(float(acc), 4),
        "macro_precision": round(float(macro_p), 4),
        "macro_recall": round(float(macro_r), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "per_intent": report_dict
    }
    
    output_path = os.path.join(RESULTS_DIR, "majority_baseline_metrics.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
        
    print(f"Majority Baseline Metrics:")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Macro F1:    {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1: {metrics['weighted_f1']:.4f}")
    
    return metrics


if __name__ == "__main__":
    run_majority_baseline()
