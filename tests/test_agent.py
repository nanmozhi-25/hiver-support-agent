"""
Unit Tests Suite for Hiver AI Support Agent.
Tests preprocessing, intent classifier, retrieval, escalation engine, end-to-end pipeline, and edge cases.
"""

import pytest
import os
import pandas as pd

from src.preprocessing import clean_text, conversation_level_split
from src.intent_classifier import IntentClassifier
from src.retrieval import ResolutionRetriever
from src.escalation import EscalationEngine
from src.pipeline import SupportAgentPipeline


def test_preprocessing_clean_text():
    raw_text = "  @AmazonHelp   Where is my order #12345?  https://t.co/abc1234  "
    cleaned = clean_text(raw_text)
    assert "@AmazonHelp" not in cleaned
    assert "https://" not in cleaned
    assert cleaned == "Where is my order #12345?"


def test_preprocessing_empty_and_invalid():
    assert clean_text("") == ""
    assert clean_text(None) == ""
    assert clean_text(12345) == ""


def test_conversation_level_split_no_leakage():
    data = {
        "conversation_id": ["C1", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"],
        "cleaned_customer_text": ["text"] * 10,
        "cleaned_brand_text": ["reply"] * 10
    }
    df = pd.DataFrame(data)
    train_df, dev_df, test_df = conversation_level_split(df, 0.6, 0.2, 0.2, random_seed=42)
    
    train_ids = set(train_df["conversation_id"])
    dev_ids = set(dev_df["conversation_id"])
    test_ids = set(test_df["conversation_id"])
    
    # Assert zero conversation overlap
    assert len(train_ids.intersection(dev_ids)) == 0
    assert len(train_ids.intersection(test_ids)) == 0
    assert len(dev_ids.intersection(test_ids)) == 0


def test_intent_classifier_output_schema():
    classifier = IntentClassifier()
    res = classifier.predict("Where is my package? The tracking number is not updating.")
    assert "intent" in res
    assert "confidence" in res
    assert "all_probabilities" in res
    assert 0.0 <= res["confidence"] <= 1.0
    assert isinstance(res["intent"], str)


def test_retrieval_output_schema():
    retriever = ResolutionRetriever()
    res = retriever.search("My order hasn't arrived yet", top_k=2)
    assert "results" in res
    assert len(res["results"]) <= 2
    if res["results"]:
        top = res["results"][0]
        assert "similarity" in top
        assert "customer_message" in top
        assert "historical_response" in top
        assert "resolution" in top


def test_escalation_sensitive_keyword():
    engine = EscalationEngine()
    res = engine.evaluate(
        customer_text="My account was hacked and stolen",
        intent="account_access",
        intent_confidence=0.99,
        retrieval_results=[{"similarity": 0.95}]
    )
    assert res["action"] == "ESCALATE"
    assert "sensitive keyword" in res["reason"].lower()


def test_escalation_short_message():
    engine = EscalationEngine()
    res = engine.evaluate(
        customer_text="help refund",
        intent="refund_request",
        intent_confidence=0.95,
        retrieval_results=[{"similarity": 0.90}]
    )
    assert res["action"] == "ESCALATE"
    assert "short" in res["reason"].lower()


def test_pipeline_end_to_end():
    pipeline = SupportAgentPipeline()
    res = pipeline.process("Where is my refund?")
    
    assert res["brand"] == "AmazonHelp"
    assert "intent" in res
    assert "intent_confidence" in res
    assert "retrieved_evidence" in res
    assert "draft_reply" in res
    assert res["action"] in ["AUTO_HANDLE", "ESCALATE"]
    assert "decision_reason" in res
