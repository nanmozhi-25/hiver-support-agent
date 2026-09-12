"""
End-to-End Customer Support Pipeline CLI & Python API.
Executes complete pipeline: Query -> Intent -> Retrieval -> Evidence -> Reply -> Escalation Decision.
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Any

from src.intent_classifier import IntentClassifier
from src.retrieval import ResolutionRetriever
from src.reply_generator import ReplyGenerator
from src.escalation import EscalationEngine


class SupportAgentPipeline:
    """
    Unified end-to-end pipeline for AI support agent.
    """
    def __init__(self, brand: str = "AmazonHelp"):
        self.brand = brand
        self.intent_classifier = IntentClassifier()
        self.retriever = ResolutionRetriever()
        self.reply_generator = ReplyGenerator()
        self.escalation_engine = EscalationEngine()

    def process(self, customer_text: str) -> Dict[str, Any]:
        """
        Processes a single customer message through the end-to-end support pipeline.
        """
        # 1. Intent Classification
        intent_res = self.intent_classifier.predict(customer_text)
        predicted_intent = intent_res["intent"]
        intent_confidence = intent_res["confidence"]

        # 2. Historical Retrieval
        retrieval_res = self.retriever.search(customer_text, top_k=3)
        retrieved_evidence = retrieval_res.get("results", [])

        # 3. Grounded Reply Generation
        draft_reply = self.reply_generator.generate(customer_text, predicted_intent, retrieved_evidence)

        # 4. Escalation Decision Engine
        escalation_res = self.escalation_engine.evaluate(
            customer_text=customer_text,
            intent=predicted_intent,
            intent_confidence=intent_confidence,
            retrieval_results=retrieved_evidence
        )

        return {
            "brand": self.brand,
            "intent": predicted_intent,
            "intent_confidence": intent_confidence,
            "retrieved_evidence": retrieved_evidence,
            "draft_reply": draft_reply,
            "action": escalation_res["action"],
            "decision_reason": escalation_res["reason"],
            "evidence_strength": escalation_res["evidence_strength"]
        }


def format_cli_output(result: Dict[str, Any]) -> str:
    """
    Formats pipeline dictionary output into clean human-readable CLI report.
    """
    evidence_lines = []
    for idx, item in enumerate(result["retrieved_evidence"], 1):
        evidence_lines.append(f"  {idx}. (Similarity: {item['similarity']:.2f}) Res: {item['resolution']}")

    evidence_str = "\n".join(evidence_lines) if evidence_lines else "  No relevant historical evidence found."

    output = f"""
=====================================================
            AI SUPPORT AGENT PIPELINE RESULT         
=====================================================
Brand:             {result['brand']}
Intent:             {result['intent']}
Confidence:         {result['intent_confidence']:.4f}

Evidence:
{evidence_str}

Draft Reply:
"{result['draft_reply']}"

Decision:           {result['action']}
Reason:             {result['decision_reason']}
Evidence Strength:  {result['evidence_strength']:.4f}
=====================================================
"""
    return output


def main():
    parser = argparse.ArgumentParser(description="Run End-to-End AI Support Agent Pipeline")
    parser.add_argument("--text", type=str, required=True, help="Customer message text to process")
    parser.add_argument("--json", action="store_true", help="Output raw JSON format instead of human-readable text")
    args = parser.parse_args()

    pipeline = SupportAgentPipeline()
    result = pipeline.process(args.text)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(format_cli_output(result))


if __name__ == "__main__":
    main()
