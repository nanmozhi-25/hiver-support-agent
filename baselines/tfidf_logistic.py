"""
Baseline 2 — TF-IDF + Logistic Regression Classifier.
Trains a TF-IDF vectorizer and calibrated Logistic Regression on conversation-level train data.
Evaluates on the Golden Set and outputs baseline metrics + confusion matrix figure.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, confusion_matrix

TRAIN_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "train.csv")
GOLDEN_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "golden", "golden_set.csv")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")


def train_and_eval_tfidf_logistic(train_csv: str = TRAIN_CSV, golden_csv: str = GOLDEN_CSV) -> dict:
    df_train = pd.read_csv(train_csv)
    df_golden = pd.read_csv(golden_csv)
    
    # Clean texts
    X_train = df_train["cleaned_customer_text"].fillna("").tolist()
    
    # We assign pseudo training labels using golden rule matcher or train intent assignments
    from eval.build_golden_set import assign_intent_and_action
    y_train = [assign_intent_and_action(text)[0] for text in X_train]
    
    X_golden = df_golden["text"].fillna("").tolist()
    y_golden = df_golden["true_intent"].tolist()
    
    # Fit TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(max_features=2500, ngram_range=(1, 2), stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_golden_vec = vectorizer.transform(X_golden)
    
    # Train Logistic Regression
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    clf.fit(X_train_vec, y_train)
    
    # Predict on Golden Set
    y_pred = clf.predict(X_golden_vec)
    y_proba = clf.predict_proba(X_golden_vec)
    
    # Metrics
    acc = accuracy_score(y_golden, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_golden, y_pred, average="macro", zero_division=0)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(y_golden, y_pred, average="weighted", zero_division=0)
    
    report_dict = classification_report(y_golden, y_pred, output_dict=True, zero_division=0)
    
    metrics = {
        "model": "TF-IDF + Logistic Regression",
        "accuracy": round(float(acc), 4),
        "macro_precision": round(float(macro_p), 4),
        "macro_recall": round(float(macro_r), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "per_intent": report_dict
    }
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    with open(os.path.join(RESULTS_DIR, "baseline_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
        
    # Plot Confusion Matrix
    labels = sorted(list(set(y_golden) | set(y_pred)))
    cm = confusion_matrix(y_golden, y_pred, labels=labels)
    
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Baseline 2: TF-IDF + Logistic Regression Confusion Matrix")
    plt.xlabel("Predicted Intent")
    plt.ylabel("True Intent")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "tfidf_baseline_cm.png"))
    plt.close()
    
    print(f"TF-IDF + Logistic Regression Baseline Metrics:")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Macro F1:    {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"Saved confusion matrix figure to {os.path.join(FIGURES_DIR, 'tfidf_baseline_cm.png')}")
    
    return metrics, clf, vectorizer


if __name__ == "__main__":
    train_and_eval_tfidf_logistic()
