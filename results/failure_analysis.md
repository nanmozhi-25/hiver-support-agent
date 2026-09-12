# Failure Analysis — AI Customer Support Agent

This document analyzes specific edge cases, failure modes, and systematic errors observed during evaluation of the AmazonHelp AI support pipeline on the Golden Evaluation set (`golden_set.csv`).

---

## Failure Mode 1: Multi-Intent Query Blending (Delivery Delay + Refund Demand)

- **Failure Category**: Multi-intent message
- **Real Example**: `"My order #112-9843102 says delivered 3 days ago but nothing arrived! I want my money refunded immediately."`
- **Expected Result**: 
  - Intent: `refund_request` (or multi-intent routing)
  - Action: `AUTO_HANDLE` / `ESCALATE` with dual resolution
- **Actual Result**: 
  - Intent: `delivery_delay` (Confidence: 0.81)
  - Action: `AUTO_HANDLE`
  - Draft Reply: *"Hi! Please DM us your order number and full delivery address so we can check the latest tracking status."*
- **Why the Failure Happened**: The single-label classifier prioritized the tracking/delivery keywords (`"delivered"`, `"nothing arrived"`) over the monetary refund request (`"money refunded"`), leading to a single-intent classification that ignored the customer's financial demand.
- **Hypothesis**: Single-label classification architectures struggle when customers express multiple distinct intents (logistics complaint + monetary request) in one message.
- **Possible Fix**: Implement multi-label intent classification or a sequence tagger that extracts multiple intent tags and composes a composite reply or routes to escalation when multi-intent conflicts arise.

---

## Failure Mode 2: Extreme Short Message Ambiguity (< 4 words)

- **Failure Category**: Noisy/short tweet
- **Real Example**: `"help refund"`
- **Expected Result**: 
  - Intent: `refund_request`
  - Action: `ESCALATE` (Reason: Message lacks order ID or specific context)
- **Actual Result**: 
  - Intent: `refund_request` (Confidence: 0.62)
  - Action: `AUTO_HANDLE` (when threshold was set at 0.60)
  - Draft Reply: *"Hello! To check on your refund status, please DM us your order ID."*
- **Why the Failure Happened**: The intent classifier detected `"refund"`, but because the message contained no order number, email, or context, auto-replying without human review risks frustrating an already unhappy customer.
- **Hypothesis**: Intent confidence alone is an insufficient safeguard for short queries (< 4 words) without strict word-count constraints.
- **Possible Fix**: Enforce a mandatory word-count threshold in `config/escalation_policy.yaml` that automatically escalates any query under 4 words regardless of classifier confidence.

---

## Failure Mode 3: Low Vector Retrieval Similarity for Rare Edge Cases

- **Failure Category**: Poor retrieval / Insufficient historical evidence
- **Real Example**: `"Can I use an Amazon UK gift card code on the US store for a Prime Video rental?"`
- **Expected Result**: 
  - Intent: `payment_billing`
  - Action: `ESCALATE` (Reason: Cross-region gift card policy requires explicit agent confirmation)
- **Actual Result**: 
  - Intent: `digital_prime_issue` (Confidence: 0.58)
  - Retrieval Similarity: 0.41 (Top match: generic Prime Video streaming error)
  - Action: `ESCALATE` (Escalation triggered by low retrieval similarity)
- **Why the Failure Happened**: The historical training corpus contained few cross-region currency conversion examples, causing low vector retrieval similarity (< 0.50).
- **Hypothesis**: Sparse representation in training corpus leads to low retrieval similarity scores.
- **Possible Fix**: Expand historical retrieval corpus with explicit knowledge base (KB) documentation and cross-region policy articles to supplement raw Twitter support threads.

---

## Failure Mode 4: False Escalation on Standard Account Login Requests

- **Failure Category**: Incorrect escalation (False Positive Escalation)
- **Real Example**: `"I forgot my password and I am not getting the reset email to my inbox."`
- **Expected Result**: 
  - Intent: `account_access`
  - Action: `AUTO_HANDLE`
- **Actual Result**: 
  - Intent: `account_access` (Confidence: 0.68)
  - Retrieval Similarity: 0.54
  - Action: `ESCALATE`
  - Reason: *"Retrieved historical match similarity (0.54) is below minimum evidence threshold (0.60)."*
- **Why the Failure Happened**: Historical Twitter agents frequently used varied phrasing for account recovery (e.g. sharing direct URL links vs asking for DMs), reducing cosine similarity between phrasing variations.
- **Hypothesis**: Strict retrieval similarity threshold (0.60) causes false escalations on benign, standard inquiries when phrasing varies.
- **Possible Fix**: Tune escalation thresholds per intent on the `dev.csv` split, allowing lower retrieval similarity thresholds for safe self-service intents like `account_access`.

---

## Failure Mode 5: Misclassified Defective Item as General Delivery Query

- **Failure Category**: Ambiguous intent / Keyword overlap
- **Real Example**: `"The package arrived on time but the box was torn open and the headset inside is completely broken."`
- **Expected Result**: 
  - Intent: `damaged_defective`
  - Action: `AUTO_HANDLE` (Prompt for photos / DM)
- **Actual Result**: 
  - Intent: `delivery_delay` (Confidence: 0.53)
  - Action: `ESCALATE`
- **Why the Failure Happened**: The presence of `"package arrived on time"` triggered strong weights for delivery tracking features, overshadowing `"torn open"` and `"broken"`.
- **Hypothesis**: Linear n-gram TF-IDF models weigh positive delivery arrival keywords heavily, missing physical damage context.
- **Possible Fix**: Transition from TF-IDF n-gram vectors to fine-tuned transformer sentence embeddings (`all-MiniLM-L6-v2` or `bge-small-en`) to capture deep semantic intent beyond literal word matching.
