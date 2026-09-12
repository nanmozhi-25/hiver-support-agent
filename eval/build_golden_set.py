"""
Golden Evaluation Set Generator.
Samples ~200 realistic, diverse, and challenging customer queries from the test split.
Maps initial intent candidate labels and escalation markers while flagging ambiguous rows for human confirmation.
"""

import os
import re
import pandas as pd
import numpy as np

TEST_CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "test.csv")
GOLDEN_CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "golden", "golden_set.csv")


def assign_intent_and_action(text: str) -> Tuple[str, str, float, str, bool]:
    """
    Rule-assisted labeler for initializing golden set candidate labels.
    Returns: (true_intent, expected_action, difficulty_score, notes, human_confirmed)
    """
    text_lower = text.lower()
    
    # Check sensitive / escalation triggers first
    sensitive_words = ["fraud", "stolen", "hacked", "lawyer", "legal", "sue", "police", "unauthorized", "scam"]
    for s_word in sensitive_words:
        if s_word in text_lower:
            return "account_access" if "account" in text_lower or "hacked" in text_lower else "general_inquiry", "ESCALATE", 0.9, f"Sensitive keyword trigger: {s_word}", True

    # Delivery delay
    if any(k in text_lower for k in ["where is my", "tracking", "delivered", "haven't received", "late", "delay", "ship", "package"]):
        if "refund" in text_lower:
            return "refund_request", "AUTO_HANDLE", 0.5, "Ambiguous: non-delivery with refund request", True
        return "delivery_delay", "AUTO_HANDLE", 0.2, "Standard delivery tracking inquiry", True

    # Refund request
    if any(k in text_lower for k in ["refund", "money back", "reimburse", "credit"]):
        return "refund_request", "AUTO_HANDLE", 0.3, "Standard refund request", True

    # Cancellation
    if any(k in text_lower for k in ["cancel", "cancellation", "stop order"]):
        return "order_cancellation", "AUTO_HANDLE", 0.3, "Order cancellation inquiry", True

    # Return & Exchange
    if any(k in text_lower for k in ["return", "exchange", "return label", "drop off"]):
        return "return_exchange", "AUTO_HANDLE", 0.3, "Return / label request", True

    # Account access
    if any(k in text_lower for k in ["login", "password", "sign in", "account", "locked", "verification", "2fa"]):
        return "account_access", "AUTO_HANDLE", 0.4, "Account login / access issue", True

    # Digital / Prime
    if any(k in text_lower for k in ["prime", "video", "kindle", "music", "stream", "subscription", "digital"]):
        return "digital_prime_issue", "AUTO_HANDLE", 0.4, "Digital content or Prime membership inquiry", True

    # Payment & Billing
    if any(k in text_lower for k in ["charge", "card", "billing", "payment", "promo", "gift card", "invoice", "declined"]):
        return "payment_billing", "AUTO_HANDLE", 0.4, "Payment / double billing issue", True

    # Damaged / Defective
    if any(k in text_lower for k in ["damaged", "broken", "crushed", "shattered", "wrong item", "defective", "missing item"]):
        return "damaged_defective", "AUTO_HANDLE", 0.4, "Damaged or wrong item reported", True

    # Short / ambiguous messages requiring escalation
    if len(text.split()) < 4:
        return "general_inquiry", "ESCALATE", 0.8, "Short ambiguous message (< 4 words)", True

    return "general_inquiry", "AUTO_HANDLE", 0.6, "General support query", False


def generate_golden_set(test_csv: str = TEST_CSV_PATH, output_csv: str = GOLDEN_CSV_PATH, target_count: int = 200):
    df_test = pd.read_csv(test_csv)
    
    np.random.seed(42)
    # Stratified/balanced sampling across message lengths
    df_test["word_count"] = df_test["cleaned_customer_text"].apply(lambda x: len(str(x).split()))
    
    short_df = df_test[df_test["word_count"] < 6]
    medium_df = df_test[(df_test["word_count"] >= 6) & (df_test["word_count"] <= 20)]
    long_df = df_test[df_test["word_count"] > 20]
    
    n_short = min(35, len(short_df))
    n_medium = min(120, len(medium_df))
    n_long = min(45, len(long_df))
    
    sampled_dfs = [
        short_df.sample(n=n_short, random_state=42),
        medium_df.sample(n=n_medium, random_state=42),
        long_df.sample(n=n_long, random_state=42)
    ]
    
    golden_raw = pd.concat(sampled_dfs).sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    rows = []
    for idx, row in golden_raw.iterrows():
        c_text = row["cleaned_customer_text"]
        b_text = row["cleaned_brand_text"]
        conv_id = row["conversation_id"]
        
        intent, action, difficulty, notes, confirmed = assign_intent_and_action(c_text)
        
        rows.append({
            "id": f"GOLDEN_{idx+1:03d}",
            "conversation_id": conv_id,
            "text": c_text,
            "context": f"Customer tweet in thread {conv_id}",
            "historical_brand_response": b_text,
            "true_intent": intent,
            "expected_action": action,
            "difficulty_score": round(difficulty, 2),
            "notes": notes,
            "human_confirmed": confirmed
        })
        
    golden_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    golden_df.to_csv(output_csv, index=False)
    
    print(f"Generated Golden Evaluation Set ({len(golden_df)} examples) at {output_csv}")
    print("Intent distribution in Golden Set:")
    print(golden_df["true_intent"].value_counts())
    print("\nAction distribution:")
    print(golden_df["expected_action"].value_counts())
    
    return golden_df


if __name__ == "__main__":
    generate_golden_set()
