"""
Intent Classifier Evaluation Harness.
Evaluates Main Intent Classifier vs Majority and TF-IDF Baselines on the Golden Set.
Generates metrics and comparative figures.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, confusion_matrix

from src.intent_classifier import IntentClassifier

GOLDEN_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "golden", "golden_set.csv")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")


def evaluate_intent_classifier(golden_csv: str = GOLDEN_CSV) -> dict:
    df_golden = pd.read_csv(golden_csv)
    
    classifier = IntentClassifier()
    
    y_true = df_golden["true_intent"].tolist()
    texts = df_golden["text"].fillna("").tolist()
    
    predictions = classifier.predict_batch(texts)
    y_pred = [p["intent"] for p in predictions]
    confidences = [p["confidence"] for p in predictions]
    
    acc = accuracy_score(y_true, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    
    report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    metrics = {
        "model": "Main Intent Classifier (Calibrated TF-IDF Logistic)",
        "accuracy": round(float(acc), 4),
        "macro_precision": round(float(macro_p), 4),
        "macro_recall": round(float(macro_r), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "mean_confidence": round(float(np.mean(confidences)), 4),
        "per_intent": report_dict
    }
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    # Save confusion matrix figure
    labels = sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=labels, yticklabels=labels)
    plt.title("Main Intent Classifier Confusion Matrix")
    plt.xlabel("Predicted Intent")
    plt.ylabel("True Intent")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "main_classifier_cm.png"))
    plt.close()
    
    print("Main Intent Classifier Metrics:")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Macro F1:    {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1: {metrics['weighted_f1']:.4f}")
    
    return metrics


if __name__ == "__main__":
    evaluate_intent_classifier()
