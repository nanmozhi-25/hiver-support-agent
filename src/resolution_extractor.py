"""
Resolution Evidence Extractor Module.
Extracts concise, actionable support resolution points from historical multi-turn customer support interactions.
"""

import re
from typing import Dict, List, Any


def extract_resolution_summary(brand_text: str, customer_text: str = "") -> str:
    """
    Analyzes historical brand response text to extract core resolution actions
    (e.g., DM request, refund authorization, tracking link, troubleshooting step).
    """
    if not isinstance(brand_text, str) or len(brand_text.strip()) == 0:
        return "No resolution action provided."

    text_lower = brand_text.lower()
    extracted_actions = []

    # 1. Direct Message / Account details request
    if any(k in text_lower for k in ["dm", "direct message", "send us a message", "pm us", "reach out via dm"]):
        if any(k in text_lower for k in ["order number", "order #", "email", "account details", "zip code", "address"]):
            extracted_actions.append("Requested customer to DM order/account details for secure verification.")
        else:
            extracted_actions.append("Requested customer to transition to Direct Message for private support.")

    # 2. Refund / Reimbursement action
    if any(k in text_lower for k in ["processed a refund", "issued a refund", "refund has been", "credit applied"]):
        extracted_actions.append("Confirmed monetary refund has been authorized/processed.")
    elif "refund" in text_lower:
        extracted_actions.append("Provided refund status policy / processing timeframe.")

    # 3. Tracking / Shipping update
    if any(k in text_lower for k in ["tracking", "shipped", "carrier", "delivery status", "in transit"]):
        extracted_actions.append("Provided order tracking update or carrier shipment link.")

    # 4. Replacement / Exchange
    if any(k in text_lower for k in ["replacement", "reship", "sending a new", "exchange"]):
        extracted_actions.append("Offered reshipment or replacement for defective/missing item.")

    # 5. Technical troubleshooting / self-service link
    if any(k in text_lower for k in ["link", "click here", "visit", "help page", "app settings", "restart", "clear cache"]):
        extracted_actions.append("Provided official self-service help link or troubleshooting steps.")

    # 6. Cancellation
    if any(k in text_lower for k in ["cancelled", "cancellation confirmed", "order has been cancelled"]):
        extracted_actions.append("Confirmed order cancellation.")

    if not extracted_actions:
        # Fallback to cleaned concise brand text (first 2 sentences)
        sentences = [s.strip() for s in re.split(r'[.!?]', brand_text) if len(s.strip()) > 5]
        extracted_actions.append(" ".join(sentences[:2]) if sentences else brand_text[:120])

    return " | ".join(extracted_actions)


def format_evidence_block(retrieved_results: List[Dict[str, Any]]) -> str:
    """
    Formats top-k retrieved historical examples into a clean prompt evidence context block.
    """
    if not retrieved_results:
        return "No historical resolution evidence found."

    lines = []
    for idx, item in enumerate(retrieved_results, 1):
        sim = item.get("similarity", 0.0)
        c_msg = item.get("customer_message", "")
        b_msg = item.get("historical_response", "")
        res = item.get("resolution", "")
        
        lines.append(f"Example #{idx} (Similarity: {sim:.2f}):")
        lines.append(f"  Historical Customer Query: \"{c_msg}\"")
        lines.append(f"  Historical Brand Response: \"{b_msg}\"")
        lines.append(f"  Extracted Resolution Action: {res}")
        lines.append("")

    return "\n".join(lines).strip()


if __name__ == "__main__":
    sample_brand_msg = "Please DM us your order number and email address so we can check on your refund status!"
    print("Extracted Resolution Summary:")
    print(extract_resolution_summary(sample_brand_msg))
