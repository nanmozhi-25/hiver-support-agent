"""
Historical Resolution Retrieval Module.
Builds vector index over historical customer-support conversations from the training corpus.
Retrieves top-k relevant historical resolutions with similarity scoring.
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing import clean_text
from src.resolution_extractor import extract_resolution_summary

TRAIN_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "train.csv")
INDEX_CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", ".cache", "retrieval_index.pkl")


class ResolutionRetriever:
    """
    Retrieves top-k historical support resolutions using TF-IDF cosine vector index.
    """
    def __init__(self, train_csv: str = TRAIN_CSV):
        self.train_csv = train_csv
        self.vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 3), stop_words="english", sublinear_tf=True)
        self.doc_matrix = None
        self.corpus_df = None
        self.is_indexed = False

    def build_index(self):
        """
        Builds vector index from historical training corpus.
        """
        if not os.path.exists(self.train_csv):
            raise FileNotFoundError(f"Training corpus file not found at {self.train_csv}")

        print(f"Building retrieval vector index from {self.train_csv}...")
        self.corpus_df = pd.read_csv(self.train_csv)
        
        # Ensure cleaned customer text
        texts = self.corpus_df["cleaned_customer_text"].fillna("").tolist()
        self.doc_matrix = self.vectorizer.fit_transform(texts)
        self.is_indexed = True

        # Precompute resolution summaries
        if "resolution_summary" not in self.corpus_df.columns:
            self.corpus_df["resolution_summary"] = [
                extract_resolution_summary(b_msg, c_msg)
                for b_msg, c_msg in zip(self.corpus_df["brand_text"], self.corpus_df["customer_text"])
            ]

        self._save_index()
        print(f"Retriever index built successfully with {len(self.corpus_df)} historical conversations.")

    def search(self, query: str, top_k: int = 3, intent_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieves top-k historical customer-support conversations matching the query.
        """
        if not self.is_indexed:
            self._load_or_build_index()

        cleaned_query = clean_text(query)
        if not cleaned_query:
            return {"results": []}

        query_vec = self.vectorizer.transform([cleaned_query])
        similarities = cosine_similarity(query_vec, self.doc_matrix)[0]

        # Top k indices
        top_indices = np.argsort(similarities)[::-1][:top_k * 2]  # fetch 2k candidates for filtering

        results = []
        for idx in top_indices:
            sim = float(np.round(similarities[idx], 4))
            row = self.corpus_df.iloc[idx]

            results.append({
                "similarity": sim,
                "conversation_id": str(row.get("conversation_id", f"CONV_{idx}")),
                "customer_message": str(row.get("customer_text", "")),
                "historical_response": str(row.get("brand_text", "")),
                "resolution": str(row.get("resolution_summary", extract_resolution_summary(row.get("brand_text", ""))))
            })

            if len(results) >= top_k:
                break

        return {"results": results}

ARTIFACT_INDEX_PATH = os.path.join(os.path.dirname(__file__), "..", "artifacts", "retrieval_index.pkl")

    def _save_index(self):
        for path in [ARTIFACT_INDEX_PATH, INDEX_CACHE_PATH]:
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "wb") as f:
                    pickle.dump({"vectorizer": self.vectorizer, "doc_matrix": self.doc_matrix, "corpus_df": self.corpus_df}, f)
                break
            except Exception:
                pass  # Ignore read-only filesystem errors on Vercel

    def _load_or_build_index(self):
        for path in [ARTIFACT_INDEX_PATH, INDEX_CACHE_PATH]:
            if os.path.exists(path):
                try:
                    with open(path, "rb") as f:
                        data = pickle.load(f)
                        self.vectorizer = data["vectorizer"]
                        self.doc_matrix = data["doc_matrix"]
                        self.corpus_df = data["corpus_df"]
                        self.is_indexed = True
                        return
                except Exception:
                    pass
        self.build_index()


if __name__ == "__main__":
    retriever = ResolutionRetriever()
    res = retriever.search("My order hasn't arrived yet and tracking is not updating", top_k=3)
    print("Top Retrieval Results:")
    for r in res["results"]:
        print(f"  Similarity: {r['similarity']:.4f} | Customer: '{r['customer_message']}'")
        print(f"  Resolution: {r['resolution']}\n")
