# Hiver AI Customer Support Agent & Evaluation Suite

Production-quality AI customer-support agent and evaluation framework built for the **Hiver SDE Intern Take-Home Assignment**. Based on historical customer support interactions from the Twitter Customer Support dataset (`twcs.csv`), focused on **AmazonHelp**.

---

## 1. Project Overview
This repository implements an end-to-end AI support agent that:
1. Classifies incoming customer support messages into data-derived intents (`config/intents.yaml`).
2. Retrieves historical resolution evidence showing how human agents resolved similar issues (`src/retrieval.py`).
3. Generates grounded support replies anchored in historical evidence (`src/reply_generator.py`).
4. Makes explainable auto-handle vs. escalation decisions (`src/escalation.py`).
5. Evaluates system performance across intent accuracy, retrieval coverage, safety risks, LLM-as-judge rubric, and human agreement (`evaluation/run_all.py`).

---

## 2. Problem Statement
Automating customer support requires guaranteeing that AI-generated responses are truthful, grounded in company resolution history, and safely escalated when inquiries are ambiguous, low-confidence, or sensitive.

---

## 3. Selected Brand: AmazonHelp
After analyzing 2.8M tweets across 30 major brands in `twcs.csv` (`data/brand_analysis.csv`), **AmazonHelp** was selected based on:
- **Volume**: 100,503 reconstructed customer support conversations (#1 in dataset).
- **Text Richness**: Average message length of 19.26 words.
- **Data Quality**: Low noise (< 1.3% duplicate text, 0 missing values).
- **Domain Diversity**: Covers logistics, refunds, returns, Prime media, and billing.

---

## 4. System Architecture
```
[Customer Message] ──► Intent Classifier ──► Vector Retriever ──► Evidence Extractor
                                                                        │
[CLI / API Result] ◄── Escalation Engine ◄── Grounded Reply Generator ◄─┘
```

---

## 5. Quick-Start Guide (Execution < 15 Minutes)

### Prerequisites
- Python 3.10+ installed
- Git

### Installation
```bash
# 1. Clone repository
git clone https://github.com/example/hiver-support-agent.git
cd hiver-support-agent

# 2. Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Environment Setup
Copy the environment template:
```bash
cp .env.example .env
```
*(Note: The system includes a deterministic mock mode, so no live API keys are required for full execution).*

### 1-Command Data Preparation & Master Evaluation
```bash
# 1. Reconstruct AmazonHelp conversations and create splits
python -m src.prepare_data --brand AmazonHelp --sample-size 5000

# 2. Run interactive Web App Server & Dashboard (Host Link: http://localhost:8000)
python -m src.web_server

# 3. Run single CLI query example
python -m src.pipeline --text "Where is my refund?"

# 4. Run complete master evaluation suite (< 15 seconds)
python -m evaluation.run_all --quick

# 5. Run unit test suite
pytest tests/
```

---

## 6. Golden Evaluation Set
Located at `data/golden/golden_set.csv` (documentation in `data/golden/README.md`):
- **Size**: 200 manually annotated examples derived from the held-out `test.csv` split.
- **Stratified Sampling**: 35 short (< 6 words), 120 medium (6–20 words), and 45 long (> 20 words) messages.
- **Data Isolation**: Golden set is strictly isolated from model training, retrieval indices, and threshold tuning (`dev.csv`).

---

## 7. Empirical Results Summary

| Model / System | Accuracy | Macro F1 | Weighted F1 | Headline Metric Notes |
| :--- | ---: | ---: | ---: | :--- |
| **Majority Baseline** | 73.50% | 9.41% | 62.27% | Predicts `general_inquiry` majority class |
| **TF-IDF + Logistic Regression** | 91.50% | 84.29% | 91.74% | Uncalibrated linear baseline |
| **Proposed System** | **93.50%** | **67.38%** | **92.47%** | **Calibrated classifier + 0.00% False Auto-Handle Rate** |

### Additional Component Metrics
- **Retrieval Coverage (Sim >= 0.50)**: 26.50%
- **False Auto-Handle Rate**: **0.00%** (Zero unsafe automated replies!)
- **LLM Judge Response Quality**: **3.72 / 4.00**
- **Human vs LLM Judge Exact Agreement**: **80.00%** (100.00% within 1 point)

---

## 8. Failure Analysis & Key Takeaways
See `results/failure_analysis.md` for in-depth breakdown of 5 failure modes:
1. Multi-intent query blending (delivery delay + refund demand).
2. Short message ambiguity (< 4 words).
3. Sparse retrieval coverage for rare edge cases.
4. False escalation on standard phrasing variations.
5. Keyword overlap misclassification.

---

## 9. Decision Log
See `decision_log.md` for 12 non-obvious engineering decisions covering brand choice, conversation-level splitting, escalation guardrails, and metric priorities.

---

## 10. Technical Report
See `report/report.md` for the complete 6-page technical report, including the mandatory section:
**"WHAT IS MISLEADING ABOUT MY HEADLINE NUMBER?"**

---

## 11. Citations & References
- **Dataset**: Customer Support on Twitter (`thoughtvector/customer-support-on-twitter` on Kaggle).
- **Libraries**: `scikit-learn`, `pandas`, `numpy`, `pyyaml`, `pytest`, `scipy`, `matplotlib`, `seaborn`.
