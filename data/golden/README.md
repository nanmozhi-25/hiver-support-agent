# Golden Evaluation Set Documentation

## Overview
The Golden Evaluation Set contains **200 manually annotated customer support messages** extracted from the held-out `test.csv` split of AmazonHelp Twitter customer interactions.

## Sampling Methodology
To ensure rigorous evaluation without selection bias, the 200 golden examples were sampled using stratified length sampling across three distinct buckets:
1. **Short Messages (< 6 words)**: 35 examples (testing handling of ambiguous, low-context tweets).
2. **Medium Messages (6–20 words)**: 120 examples (representing typical customer queries).
3. **Long Messages (> 20 words)**: 45 examples (testing complex multi-turn or multi-problem inquiries).

## Labeling Schema
Each row in `golden_set.csv` contains the following fields:
- `id`: Unique golden evaluation identifier (`GOLDEN_001` to `GOLDEN_200`).
- `conversation_id`: Original tweet thread ID from `twcs.csv`.
- `text`: Cleaned customer message text.
- `context`: Additional conversation thread metadata.
- `historical_brand_response`: Actual historical resolution reply given by AmazonHelp agents.
- `true_intent`: Ground truth intent label assigned from the 9-intent taxonomy (`config/intents.yaml`).
- `expected_action`: Expected routing decision (`AUTO_HANDLE` vs `ESCALATE`).
- `difficulty_score`: Estimated message difficulty score (0.0 = unambiguous to 1.0 = highly complex/risky).
- `notes`: Specific rationale for escalation, keyword flags, or intent ambiguity.
- `human_confirmed`: Boolean flag indicating whether the label was confirmed by human audit.

## Intent Taxonomy Summary
1. `delivery_delay` — Package tracking, late shipment, missing delivery.
2. `refund_request` — Refund inquiries, reimbursement status, money back.
3. `order_cancellation` — Cancelling active orders or unexpected cancellation explanations.
4. `return_exchange` — Return shipping labels, return window, drop-off locations.
5. `account_access` — Login errors, password reset, account verification.
6. `digital_prime_issue` — Prime Video errors, Prime billing, Kindle books, digital media.
7. `payment_billing` — Declined cards, double charges, invoice errors, promo codes.
8. `damaged_defective` — Broken items, crushed boxes, wrong item received.
9. `general_inquiry` — Store hours, policy questions, general contact inquiries.

## Escalation Criteria in Golden Set
Messages are labeled as `ESCALATE` if they meet any of the following:
- Contains high-risk/sensitive keywords (`fraud`, `stolen`, `hacked`, `lawyer`, `legal`, `sue`, `police`, `unauthorized`, `scam`).
- Highly ambiguous or short text (< 4 words) where auto-reply risks providing misleading information.
- Conflicting multi-intent requests (e.g. demanding immediate refund while reporting lost shipment without order ID).

## Strict Data Leakage & Isolation Policy
- **No Overlap**: Golden set examples are derived exclusively from the held-out `test.csv` split.
- **Zero Tuning Exposure**: The Golden Evaluation set is NEVER used during vector retrieval index construction, TF-IDF vocabulary fitting, classifier training, or threshold tuning (which exclusively uses `dev.csv`).
