"""
Escalation Engine Evaluation Harness.
Evaluates AUTO_HANDLE vs ESCALATE decisions on the Golden Set.
Computes Precision, Recall, F1, and tracks False Auto-handling Rate (dangerous safety failures).
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, confusion_matrix

from src.pipeline import SupportAgentPipeline

GOLDEN_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "golden", "golden_set.csv")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")


def evaluate_escalation_engine(golden_csv: str = GOLDEN_CSV) -> dict:
    df_golden = pd.read_csv(golden_csv)
    pipeline = SupportAgentPipeline()
    
    y_true = df_golden["expected_action"].tolist()
    y_pred = []
    reasons = []
    
    for _, row in df_golden.iterrows():
        text = str(row["text"])
        res = pipeline.process(text)
        y_pred.append(res["action"])
        reasons.append(res["decision_reason"])
        
    acc = accuracy_score(y_true, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    
    labels = ["AUTO_HANDLE", "ESCALATE"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Calculate False Auto-Handling (True ESCALATE predicted as AUTO_HANDLE)
    # cm[1][0] is True ESCALATE predicted as AUTO_HANDLE
    false_auto_handle_count = int(cm[1][0])
    false_escalation_count = int(cm[0][1])
    total_escalate_ground_truth = int(np.sum(cm[1]))
    
    false_auto_handle_rate = round(false_auto_handle_count / max(total_escalate_ground_truth, 1), 4)
    
    report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    metrics = {
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "false_auto_handle_count": false_auto_handle_count,
        "false_escalation_count": false_escalation_count,
        "false_auto_handle_rate": false_auto_handle_rate,
        "classification_report": report_dict
    }
    
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", xticklabels=labels, yticklabels=labels)
    plt.title("Escalation Engine Decision Confusion Matrix")
    plt.xlabel("Predicted Action")
    plt.ylabel("Expected Action (Golden Ground Truth)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "escalation_cm.png"))
    plt.close()
    
    print("Escalation Engine Metrics:")
    print(f"  Accuracy:                 {metrics['accuracy']:.4f}")
    print(f"  Macro F1:                 {metrics['macro_f1']:.4f}")
    print(f"  False Auto-Handle Count:  {false_auto_handle_count} (CRITICAL SAFETY RISK)")
    print(f"  False Escalation Count:   {false_escalation_count}")
    print(f"  False Auto-Handle Rate:   {false_auto_handle_rate * 100:.2f}%")
    
    return metrics


if __name__ == "__main__":
    evaluate_escalation_engine()
