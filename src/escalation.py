"""
Escalation Engine Module.
Evaluates multi-signal decision metrics (intent confidence, retrieval similarity, sensitive risk keywords)
to decide between AUTO_HANDLE and ESCALATE, providing human-readable decision explanations.
"""

import os
import yaml
import re
from typing import Dict, List, Any, Optional, Tuple

POLICY_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "escalation_policy.yaml")


class EscalationEngine:
    """
    Multi-signal escalation decision engine.
    """
    def __init__(self, policy_path: str = POLICY_PATH):
        self.policy_path = policy_path
        self.policy = self._load_policy()
        
        thresholds = self.policy.get("thresholds", {})
        self.min_intent_confidence = thresholds.get("min_intent_confidence", 0.65)
        self.min_retrieval_similarity = thresholds.get("min_retrieval_similarity", 0.60)
        
        risk_rules = self.policy.get("risk_rules", {})
        self.sensitive_keywords = risk_rules.get("sensitive_keywords", [])
        self.high_risk_intents = risk_rules.get("high_risk_intents", [])

    def _load_policy(self) -> dict:
        if os.path.exists(self.policy_path):
            with open(self.policy_path, "r") as f:
                return yaml.safe_load(f)
        return {}

    def evaluate(
        self,
        customer_text: str,
        intent: str,
        intent_confidence: float,
        retrieval_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluates input text, intent confidence, and retrieval evidence to output decision and reason.
        """
        text_lower = customer_text.lower()
        word_count = len(text_lower.split())

        # Signal 1: Check sensitive keywords (High Risk Escalation)
        for keyword in self.sensitive_keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
                return {
                    "action": "ESCALATE",
                    "reason": f"Message contains high-risk sensitive keyword trigger: '{keyword}'. Requires human specialist.",
                    "confidence": round(intent_confidence, 4),
                    "evidence_strength": 0.0
                }

        # Signal 2: Short / ambiguous message check
        if word_count < 4:
            return {
                "action": "ESCALATE",
                "reason": f"Customer message is very short ({word_count} words) and lacks sufficient context for automated resolution.",
                "confidence": round(intent_confidence, 4),
                "evidence_strength": 0.20
            }

        # Signal 3: Intent confidence threshold
        if intent_confidence < self.min_intent_confidence:
            return {
                "action": "ESCALATE",
                "reason": f"Predicted intent confidence ({intent_confidence:.2f}) is below auto-handling threshold ({self.min_intent_confidence:.2f}).",
                "confidence": round(intent_confidence, 4),
                "evidence_strength": 0.30
            }

        # Signal 4: Top retrieval similarity check
        top_similarity = retrieval_results[0]["similarity"] if retrieval_results else 0.0
        if top_similarity < self.min_retrieval_similarity:
            return {
                "action": "ESCALATE",
                "reason": f"Retrieved historical match similarity ({top_similarity:.2f}) is below minimum evidence threshold ({self.min_retrieval_similarity:.2f}).",
                "confidence": round(intent_confidence, 4),
                "evidence_strength": round(top_similarity, 4)
            }

        # Signal 5: High risk intent extra validation
        if intent in self.high_risk_intents and top_similarity < 0.75:
            return {
                "action": "ESCALATE",
                "reason": f"High-risk intent '{intent}' requires top retrieval similarity >= 0.75 (current: {top_similarity:.2f}).",
                "confidence": round(intent_confidence, 4),
                "evidence_strength": round(top_similarity, 4)
            }

        # High-confidence Auto Handle
        evidence_strength = round((intent_confidence + top_similarity) / 2.0, 4)
        return {
            "action": "AUTO_HANDLE",
            "reason": f"High intent confidence ({intent_confidence:.2f}) and strong historical evidence similarity ({top_similarity:.2f}) support automated response.",
            "confidence": round(intent_confidence, 4),
            "evidence_strength": evidence_strength
        }


if __name__ == "__main__":
    engine = EscalationEngine()
    eval_res = engine.evaluate(
        customer_text="Where is my refund?",
        intent="refund_request",
        intent_confidence=0.85,
        retrieval_results=[{"similarity": 0.78}]
    )
    print("Escalation Engine Decision:")
    print(eval_res)
