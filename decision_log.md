# Engineering Decision Log

This log documents 12 non-obvious engineering decisions, trade-offs, and architectural choices made during the development of the Hiver AI Support Agent.

---

### Decision 1: Target Brand Selection (`AmazonHelp`)
- **Decision**: Selected `AmazonHelp` over 30 other customer support handles in `twcs.csv`.
- **Reason**: `AmazonHelp` provides the highest conversation volume (100,503 reconstructed customer support conversations), highest message length (19.26 words/msg), lowest noise (1.25% duplicates, 0 missing values), and diverse e-commerce support domains.
- **Trade-off**: Requires handling large dataset volume (270k+ tweets), necessitating efficient subsampling for fast evaluation.

---

### Decision 2: Conversation-Level Train/Dev/Test Splitting
- **Decision**: Split data strictly by `conversation_id` rather than random tweet-level splitting.
- **Reason**: Prevents data leakage where messages from the same support thread appear in both training retrieval corpus and evaluation golden test set.
- **Trade-off**: Slower partitioning logic compared to naive random splitting.

---

### Decision 3: Data-Derived 9-Intent Taxonomy
- **Decision**: Defined 9 specific support intents (`delivery_delay`, `refund_request`, `order_cancellation`, `return_exchange`, `account_access`, `digital_prime_issue`, `payment_billing`, `damaged_defective`, `general_inquiry`) based on actual brand conversation frequencies.
- **Reason**: Prevents artificial or misaligned intent taxonomy that doesn't reflect real customer queries.
- **Trade-off**: Single-label taxonomy cannot natively represent complex multi-intent messages in one pass.

---

### Decision 4: Stratified Golden Evaluation Set (200 Examples)
- **Decision**: Sampled 200 golden test examples using stratified message length buckets (short < 6 words, medium 6-20 words, long > 20 words).
- **Reason**: Ensures evaluation is not biased toward easy, medium-length queries and tests edge cases like ambiguous short tweets.
- **Trade-off**: Requires dedicated annotation infrastructure and audit tracking (`human_confirmed` flag).

---

### Decision 5: Calibrated Logistic Regression Classifier
- **Decision**: Used n-gram TF-IDF paired with `CalibratedClassifierCV` (sigmoid Platt scaling) for intent classification.
- **Reason**: Provides true calibrated confidence probabilities required for escalation thresholding, while remaining lightweight and fully explainable.
- **Trade-off**: Slightly lower semantic feature representation compared to large fine-tuned transformer models.

---

### Decision 6: Historical Resolution Evidence Extraction
- **Decision**: Extracted structured resolution actions (e.g. DM request, refund authorization, tracking link) from raw multi-turn support threads before injecting into generation prompts.
- **Reason**: Prevents copying noisy conversational filler (e.g. "Hi @user, thanks for tweeting us") into LLM prompts.
- **Trade-off**: Requires heuristic parsing rules to extract key action phrases.

---

### Decision 7: Multi-Signal Escalation Engine
- **Decision**: Evaluates intent confidence, vector retrieval similarity, word count, and sensitive keyword triggers to determine `AUTO_HANDLE` vs `ESCALATE`.
- **Reason**: Single thresholding on intent confidence alone fails to catch low-similarity or sensitive security requests.
- **Trade-off**: Increases system strictness, leading to higher false escalation rates to guarantee 0% false auto-handling.

---

### Decision 8: Sensitive Keyword Safety Rules
- **Decision**: Immediate mandatory human escalation for high-risk security/legal keywords (`fraud`, `stolen`, `hacked`, `lawyer`, `legal`, `sue`, `police`, `unauthorized`).
- **Reason**: Automated responses to legal threats or compromised accounts carry immense brand and legal liability.
- **Trade-off**: Disables auto-handling for queries containing these keywords even if intent confidence is 1.0.

---

### Decision 9: Deterministic Mock LLM & Judge Fallback Mode
- **Decision**: Implemented deterministic mock generators for reply generation and LLM judge evaluation.
- **Reason**: Allows the complete repository, CLI, unit tests, and evaluation suite to run out-of-the-box without requiring active paid API keys.
- **Trade-off**: Mock replies are template-driven rather than dynamically generative.

---

### Decision 10: Prioritizing Macro F1 and False Auto-Handling Rate over Accuracy
- **Decision**: Treated Macro F1 and False Auto-Handling Rate as primary headline safety metrics rather than raw Accuracy.
- **Reason**: 73.5% of queries belong to `general_inquiry`; raw accuracy obscures failures on critical minority classes.
- **Trade-off**: Yields lower headline numbers (Macro F1 67.38% vs Accuracy 93.50%), but provides honest technical evaluation.

---

### Decision 11: 50-Sample Human Benchmark for LLM Judge Validation
- **Decision**: Validated LLM judge output against a 50-sample human-audited benchmark (`human_scores.csv`).
- **Reason**: Evaluates whether automated LLM judge scores correlate reliably with human quality assessments.
- **Trade-off**: Human sample size is limited to 50 examples due to manual audit effort.

---

### Decision 12: Reproducible 5,000-Conversation Subsampling Pipeline
- **Decision**: Provided a quick-start data preparation pipeline (`src/prepare_data.py`) subsampling 5,000 conversations for local execution.
- **Reason**: Allows reviewer to execute complete master evaluation suite in < 15 minutes on standard laptop CPU.
- **Trade-off**: Subsampled retrieval index contains fewer historical matches than full 100k corpus.
