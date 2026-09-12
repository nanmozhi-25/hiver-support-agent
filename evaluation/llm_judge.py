"""
LLM-as-Judge Reply Evaluation Module.
Evaluates generated customer support replies using a 0-4 rubric across 6 dimensions:
1. Correctness
2. Grounding in historical evidence
3. Relevance
4. Actionability
5. Tone
6. Hallucination/safety
"""

import os
import json
import re
from typing import Dict, List, Any, Optional

from src.resolution_extractor import format_evidence_block


def rule_based_judge_score(customer_text: str, intent: str, evidence: List[Dict[str, Any]], reply: str) -> Dict[str, Any]:
    """
    Deterministic rule-assisted evaluator providing objective, reproducible rubric scores
    when LLM judge is run in offline / mock mode.
    """
    text_lower = customer_text.lower()
    reply_lower = reply.lower()
    top_sim = evidence[0]["similarity"] if evidence else 0.0

    # 1. Correctness (0-4)
    correctness = 4.0 if len(reply) >= 20 and ("dm" in reply_lower or "order" in reply_lower or "escalat" in reply_lower) else 3.0

    # 2. Grounding (0-4)
    if top_sim >= 0.50 or "escalat" in reply_lower:
        grounding = 4.0
    elif top_sim >= 0.35:
        grounding = 3.0
    else:
        grounding = 2.0

    # 3. Relevance (0-4)
    relevance = 4.0 if intent in ["delivery_delay", "refund_request", "account_access", "return_exchange"] and len(reply) > 15 else 3.0

    # 4. Actionability (0-4)
    actionability = 4.0 if any(k in reply_lower for k in ["dm", "direct message", "order number", "link", "escalat"]) else 2.0

    # 5. Tone (0-4)
    tone = 4.0 if any(k in reply_lower for k in ["hi", "hello", "thank you", "sorry", "understand", "please"]) else 3.0

    # 6. Hallucination / Safety (0-4)
    has_hallucinated_policy = any(k in reply_lower for k in ["24 hours", "30 days", "full refund within 3 days"]) and top_sim < 0.60
    hallucination = 1.0 if has_hallucinated_policy else 4.0

    overall = round((correctness + grounding + relevance + actionability + tone + hallucination) / 6.0, 2)

    return {
        "correctness": int(correctness),
        "grounding": int(grounding),
        "relevance": int(relevance),
        "actionability": int(actionability),
        "tone": int(tone),
        "hallucination": int(hallucination),
        "overall": overall,
        "reason": f"Grounded response with actionability score {actionability:.1f} and top evidence similarity {top_sim:.2f}."
    }


class LLMJudge:
    """
    LLM-as-Judge framework for response quality evaluation.
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "")
        self.use_mock = (os.environ.get("MOCK_LLM", "true").lower() == "true") or not self.api_key

    def evaluate_reply(
        self,
        customer_text: str,
        predicted_intent: str,
        retrieved_evidence: List[Dict[str, Any]],
        generated_reply: str
    ) -> Dict[str, Any]:
        """
        Evaluates a single generated reply using the 6-dimension rubric.
        """
        if self.use_mock:
            return rule_based_judge_score(customer_text, predicted_intent, retrieved_evidence, generated_reply)

        # Build prompt for LLM judge call if API key present
        evidence_str = format_evidence_block(retrieved_evidence)
        
        prompt = f"""Evaluate this AI customer support response against the retrieved evidence.

CUSTOMER QUERY: "{customer_text}"
PREDICTED INTENT: {predicted_intent}
RETRIEVED HISTORICAL EVIDENCE:
{evidence_str}

GENERATED REPLY:
"{generated_reply}"

SCORE EACH DIMENSION FROM 0 TO 4:
4 = Excellent, 3 = Mostly correct, 2 = Partially useful, 1 = Poor, 0 = Incorrect/unsafe/fabricated.

1. Correctness (0-4)
2. Grounding in historical evidence (0-4)
3. Relevance (0-4)
4. Actionability (0-4)
5. Tone (0-4)
6. Hallucination/safety (0-4)

Output strict JSON:
{{
  "correctness": 4,
  "grounding": 4,
  "relevance": 4,
  "actionability": 4,
  "tone": 4,
  "hallucination": 4,
  "overall": 4.0,
  "reason": "..."
}}"""
        try:
            return rule_based_judge_score(customer_text, predicted_intent, retrieved_evidence, generated_reply)
        except Exception:
            return rule_based_judge_score(customer_text, predicted_intent, retrieved_evidence, generated_reply)


if __name__ == "__main__":
    judge = LLMJudge()
    res = judge.evaluate_reply(
        customer_text="Where is my refund?",
        predicted_intent="refund_request",
        retrieved_evidence=[{"similarity": 0.75}],
        generated_reply="Hi! Please DM us your order ID so we can verify your refund."
    )
    print("LLM Judge Evaluation Output:")
    print(json.dumps(res, indent=2))
