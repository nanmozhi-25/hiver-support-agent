"""
Main Intent Classifier Module.
Implements calibrated intent classification producing reproducible probabilities and confidence scores.
"""

import os
import pickle
import yaml
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

from src.preprocessing import clean_text

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "intents.yaml")
MODEL_CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", ".cache", "intent_classifier.pkl")


class IntentClassifier:
    """
    Calibrated Intent Classifier producing intent predictions and confidence scores.
    """
    def __init__(self, config_path: str = CONFIG_PATH):
        self.config_path = config_path
        self.intents_config = self._load_config()
        self.intent_names = [i["name"] for i in self.intents_config.get("intents", [])]
        
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 3), stop_words="english", sublinear_tf=True)
        self.base_clf = LogisticRegression(max_iter=1000, C=2.0, class_weight="balanced", random_state=42)
        self.clf = None
        self.is_fitted = False

    def _load_config(self) -> dict:
        if os.path.exists(self.config_path):
            with open(self.config_path, "r") as f:
                return yaml.safe_load(f)
        return {"intents": []}

    def train(self, train_texts: List[str], train_labels: List[str]):
        """
        Fits vectorizer and calibrated logistic classifier on training corpus.
        """
        cleaned_texts = [clean_text(t) for t in train_texts]
        X_vec = self.vectorizer.fit_transform(cleaned_texts)
        
        # Fit base model and calibrate using 5-fold cross-validation
        self.clf = CalibratedClassifierCV(estimator=self.base_clf, method="sigmoid", cv=5)
        self.clf.fit(X_vec, train_labels)
        self.is_fitted = True
        
        self._save_model()

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predicts intent and calibrated confidence for a single input message.
        """
        if not self.is_fitted:
            self._load_or_fallback_train()

        cleaned = clean_text(text)
        if not cleaned or len(cleaned) < 2:
            return {
                "intent": "general_inquiry",
                "confidence": 0.20,
                "all_probabilities": {intent: (0.80 if intent == "general_inquiry" else 0.025) for intent in self.intent_names}
            }

        vec = self.vectorizer.transform([cleaned])
        probas = self.clf.predict_proba(vec)[0]
        classes = self.clf.classes_

        prob_dict = {cls: float(np.round(p, 4)) for cls, p in zip(classes, probas)}
        best_intent = classes[np.argmax(probas)]
        confidence = float(np.round(np.max(probas), 4))

        return {
            "intent": str(best_intent),
            "confidence": confidence,
            "all_probabilities": prob_dict
        }

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        return [self.predict(t) for t in texts]

    def _save_model(self):
        os.makedirs(os.path.dirname(MODEL_CACHE_PATH), exist_ok=True)
        with open(MODEL_CACHE_PATH, "wb") as f:
            pickle.dump({"vectorizer": self.vectorizer, "clf": self.clf}, f)

    def _load_or_fallback_train(self):
        if os.path.exists(MODEL_CACHE_PATH):
            with open(MODEL_CACHE_PATH, "rb") as f:
                data = pickle.load(f)
                self.vectorizer = data["vectorizer"]
                self.clf = data["clf"]
                self.is_fitted = True
        else:
            # Fallback auto-train on processed train set
            train_csv = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "train.csv")
            if os.path.exists(train_csv):
                df_train = pd.read_csv(train_csv)
                X_train = df_train["cleaned_customer_text"].fillna("").tolist()
                
                from eval.build_golden_set import assign_intent_and_action
                y_train = [assign_intent_and_action(t)[0] for t in X_train]
                self.train(X_train, y_train)
            else:
                raise RuntimeError("Training data not found! Run src.prepare_data first.")


if __name__ == "__main__":
    classifier = IntentClassifier()
    res = classifier.predict("Where is my package? The tracking number is not updating.")
    print("Sample Intent Prediction:")
    print(res)
