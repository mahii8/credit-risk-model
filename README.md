# Credit Risk Probability Model for Alternative Data

## Project Overview
An end-to-end credit risk model for Bati Bank's buy-now-pay-later service,
using transaction data from the Xente eCommerce platform to predict customer
default probability.

## Credit Scoring Business Understanding

### 1. How does Basel II influence model interpretability?
The Basel II Capital Accord requires banks to hold capital reserves proportional
to their credit risk exposure. This means every risk model must be auditable,
interpretable, and reproducible. Regulators expect banks to explain WHY a
customer was rejected — a black-box model that cannot justify its decisions
exposes the bank to regulatory and legal risk. This drives the need for
well-documented pipelines, version-controlled data, and interpretable models
such as Logistic Regression with Weight of Evidence (WoE) encoding.

### 2. Why is a proxy variable necessary, and what risks does it introduce?
The Xente dataset contains no direct "default" label — it records transactions,
not loan repayment outcomes. A proxy variable (derived from RFM behavioral
patterns) is therefore necessary to approximate creditworthiness. Business risks
include: (1) proxy drift — the behavioral signal may not correlate with actual
default in a loan context; (2) label noise — customers labeled "high-risk" by
RFM may be perfectly creditworthy loan payers; (3) regulatory scrutiny — Basel II
requires documented justification for proxy variable design.

### 3. Trade-offs between interpretable and high-performance models
| Factor | Logistic Regression + WoE | Gradient Boosting |
|--------|--------------------------|-------------------|
| Interpretability | High — scorecard format | Low — black box |
| Performance | Moderate | High |
| Regulatory fit | Excellent | Requires SHAP/LIME |
| Auditability | Easy | Complex |
| Preferred when | Compliance-first | Performance-first |

In a regulated context like Bati Bank, a hybrid approach is recommended:
Logistic Regression for the production scorecard, Gradient Boosting for
internal benchmarking and performance validation.

## Project Structure
- notebooks/ — EDA and exploratory analysis
- src/ — Production code: feature engineering, training, API
- tests/ — Unit tests
- data/ — Raw and processed data (git-ignored)

## How to Reproduce
1. Clone the repository
2. Install dependencies: pip install -r requirements.txt
3. Download data from Kaggle: Xente Challenge
4. Place in data/raw/
5. Run: python src/data_processing.py

## Author
Mahlet Adane | 10 Academy x Kifiya | Week 4 | May 2026
