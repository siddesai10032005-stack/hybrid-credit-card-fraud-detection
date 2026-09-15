# 💳 Hybrid Two-Stage Credit Card Fraud Detection

## 🚀 Live Dashboard

👉 [Open the Live Streamlit Dashboard](https://hybrid-credit-card-fraud-detection8869.streamlit.app/)

## 🎯 Project Overview

A Hybrid Two-Stage Credit Card Fraud Detection System combining machine learning with customer behavioral verification.

The system evaluates transactions through two complementary stages:

1. **Stage 1 — Transaction-Level Fraud Detection**
2. **Stage 2 — Customer Behavioral Verification**

The outputs are combined into a hybrid risk score and converted into three operational decisions:

**APPROVE → VERIFY → BLOCK**

> ⚠️ **Dataset Note:** This project uses the publicly available Sparkov synthetic credit-card transaction dataset. It is simulated data and does not contain real banking customers or real financial transactions.

## 🏗️ System Architecture

```text
                    TRANSACTION
                         │
                         ▼
              ┌─────────────────────┐
              │ Feature Engineering │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │      STAGE 1        │
              │ Transaction-Level ML│
              │                     │
              │ Random Forest       │
              └──────────┬──────────┘
                         │
                  Fraud Probability
                         │
                         ▼
              ┌─────────────────────┐
              │      STAGE 2        │
              │ Behavioral Model    │
              │                     │
              │ Amount Behavior     │
              │ Merchant Behavior   │
              │ Category Behavior   │
              │ Time Behavior       │
              │ Location Behavior   │
              │ Velocity Behavior   │
              └──────────┬──────────┘
                         │
                  Behavior Probability
                         │
                         ▼
              ┌─────────────────────┐
              │   HYBRID SCORING    │
              │                     │
              │ 70% × Stage 1       │
              │ 30% × Stage 2       │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │   DECISION ENGINE   │
              └──────────┬──────────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
       APPROVE         VERIFY         BLOCK
       < 0.10        0.10–<0.90       ≥ 0.90

# 🔬 Stage 1 — Transaction-Level Fraud Detection

Stage 1 evaluates whether a transaction looks suspicious using transaction-level and customer-context features.

## 📌 Features Used

- Transaction amount
- Transaction hour
- Day of week
- Day
- Month
- Weekend indicator
- Customer average transaction amount
- Amount ratio
- Customer transaction count
- Location distance
- City population
- Transaction category
- Gender

## 🤖 Models Evaluated

### Logistic Regression

Used as the baseline model.

| Metric | Result |
|---|---:|
| PR-AUC | **0.1377** |
| ROC-AUC | **0.9244** |
| Recall | **77.58%** |
| Precision | **2.59%** |
| F1 Score | **0.0501** |

### Random Forest

Random Forest significantly outperformed Logistic Regression.

| Metric | Result |
|---|---:|
| PR-AUC | **0.7906** |
| ROC-AUC | **0.9959** |
| Recall | **92.45%** |
| Precision | **33.40%** |
| F1 Score | **0.4907** |

Therefore, **Random Forest was selected as the Stage 1 model**.

## 🔄 Stage 1 Output

For every transaction, Stage 1 produces a:

```text
Fraud Probability

# 🧠 Stage 2 — Customer Behavioral Verification

Stage 2 asks a different question:

> **Does this transaction look normal for this particular customer?**

Instead of evaluating only the transaction itself, the system compares the transaction against the customer's historical behavior.

## 👤 Customer Behavioral Profile

The system builds historical profiles for **983 unique customers/cards**.

Each profile contains information such as:

- Average transaction amount
- Transaction amount standard deviation
- Median transaction amount
- Maximum transaction amount
- Total transaction count
- Typical transaction hour
- Typical customer location
- Average city population
- Frequently visited merchants
- Frequently used categories

These profiles provide the behavioral reference point for Stage 2.

## 🔍 Behavioral Signals

### 💰 Amount Behavior

- Amount deviation
- Amount change from previous transaction
- Large amount jump
- Amount/location anomaly

### 🏪 Merchant Behavior

- Merchant visit frequency
- New merchant indicator
- Merchant location anomaly

### 🗂️ Category Behavior

- Category visit frequency
- New category indicator

### 🕐 Time Behavior

- Hour deviation
- Circular time difference

### 📍 Location Behavior

- Customer's usual location
- Merchant location deviation
- Location change from previous transaction

### ⚡ Velocity Behavior

- Seconds since previous transaction
- High-velocity transaction indicator
- Amount velocity anomaly

### 🔗 Combined Behavioral Anomaly

Multiple behavioral signals are combined into an overall behavioral anomaly score.

## 🤖 Behavioral Model

A **Random Forest classifier** is used to learn patterns in customer behavior.

The model uses behavioral features such as:

```text
Amount
Merchant
Category
Time
Location
Velocity
Previous Transaction Behavior
Combined Anomaly Signals


# 🔀 Hybrid Risk Scoring

The final system combines the outputs from both detection stages.

## 🧮 Hybrid Formula

```text
Hybrid Probability
=
0.70 × Stage 1 Fraud Probability
+
0.30 × Behavioral Probability

Transaction-Level Fraud Risk
             +
Customer Behavioral Risk
             ↓
     Hybrid Risk Score
             ↓
    Decision Routing

                 Hybrid Score
                      │
          ┌───────────┼───────────┐
          │           │           │
          ▼           ▼           ▼
       APPROVE      VERIFY      BLOCK
       < 0.10     0.10–<0.90    ≥ 0.90


from pathlib import Path

content = r'''
# 🚦 Decision Engine

The final hybrid score is converted into three operational decisions.

| Hybrid Score | Decision |
|---|---|
| `< 0.10` | 🟢 **APPROVE** |
| `0.10 – < 0.90` | 🟡 **VERIFY** |
| `≥ 0.90` | 🔴 **BLOCK** |

### 🟢 APPROVE

The transaction appears sufficiently normal and is allowed to proceed automatically.

### 🟡 VERIFY

The transaction has an elevated risk score but is not suspicious enough for automatic blocking.

The customer can be asked for additional verification such as:

- OTP confirmation
- Mobile-app approval
- Additional authentication
- Customer confirmation

This reduces the chance of blocking legitimate customers who make unusual purchases.

### 🔴 BLOCK

The transaction has a very high hybrid risk score and is automatically blocked.

# 📊 Final Test Results

The final hybrid system was evaluated on the chronological test dataset.

## Dataset

| Metric | Value |
|---|---:|
| Total transactions | **555,719** |
| Fraud transactions | **2,145** |
| Normal transactions | **553,574** |

## Final Decision Distribution

| Decision | Transactions |
|---|---:|
| 🟢 APPROVE | **495,711** |
| 🟡 VERIFY | **59,022** |
| 🔴 BLOCK | **986** |

## Fraud Routing

| Fraud Outcome | Count |
|---|---:|
| Fraud approved | **23** |
| Fraud verified | **1,170** |
| Fraud blocked | **952** |
| Fraud caught | **2,122** |

## Key Metrics

### 🛡️ Overall Fraud Routing Recall: 98.93%

**2,122 out of 2,145 fraudulent transactions were routed to VERIFY or BLOCK.**

Therefore:

```text
Fraud Routing Recall = 98.93%

from pathlib import Path

content = r'''
# 🚦 Decision Engine

The final hybrid score is converted into three operational decisions.

| Hybrid Score | Decision |
|---|---|
| `< 0.10` | 🟢 **APPROVE** |
| `0.10 – < 0.90` | 🟡 **VERIFY** |
| `≥ 0.90` | 🔴 **BLOCK** |

### 🟢 APPROVE

The transaction appears sufficiently normal and is allowed to proceed automatically.

### 🟡 VERIFY

The transaction has an elevated risk score but is not suspicious enough for automatic blocking.

The customer can be asked for additional verification such as:

- OTP confirmation
- Mobile-app approval
- Additional authentication
- Customer confirmation

This reduces the chance of blocking legitimate customers who make unusual purchases.

### 🔴 BLOCK

The transaction has a very high hybrid risk score and is automatically blocked.

# 📊 Final Test Results

The final hybrid system was evaluated on the chronological test dataset.

## Dataset

| Metric | Value |
|---|---:|
| Total transactions | **555,719** |
| Fraud transactions | **2,145** |
| Normal transactions | **553,574** |

## Final Decision Distribution

| Decision | Transactions |
|---|---:|
| 🟢 APPROVE | **495,711** |
| 🟡 VERIFY | **59,022** |
| 🔴 BLOCK | **986** |

## Fraud Routing

| Fraud Outcome | Count |
|---|---:|
| Fraud approved | **23** |
| Fraud verified | **1,170** |
| Fraud blocked | **952** |
| Fraud caught | **2,122** |

## Key Metrics

### 🛡️ Overall Fraud Routing Recall: 98.93%

**2,122 out of 2,145 fraudulent transactions were routed to VERIFY or BLOCK.**

Therefore:

```text
Fraud Routing Recall = 98.93%


from pathlib import Path

content = r'''
# 🔎 Error Analysis

The final system was analyzed to understand where the hybrid model performs well and where difficult fraud cases occur.

## ❌ False Positives

False positives occur when legitimate transactions receive elevated risk scores.

Common characteristics include:

- Large genuine purchases
- New merchants
- Unusual transaction times
- Unusual locations
- Sudden changes in transaction amount

This shows an important challenge in fraud detection:

> A legitimate unusual transaction can look very similar to a fraudulent transaction.

## ⚠️ False Negatives

False negatives occur when fraudulent transactions receive lower risk scores and are not routed to VERIFY or BLOCK.

These transactions can appear behaviorally normal, for example:

- Normal transaction amounts
- Familiar merchants
- Familiar locations
- Normal transaction timing
- Low behavioral anomaly scores

This is why the system uses customer behavioral verification as a second layer rather than relying only on transaction-level classification.

## 🧠 Important Behavioral Patterns

Some combinations of behavioral signals were particularly informative.

| Behavioral Pattern | Transactions | Fraud Rate |
|---|---:|---:|
| Large amount + unusual location | 5,619 | **11.03%** |
| Large amount + new merchant | 2,069 | **13.24%** |
| Large amount + unusual time | 5,138 | **22.48%** |
| Large amount + fast transaction | 279 | **34.05%** |
| New merchant + unusual location | 33,453 | **0.64%** |
| New merchant + unusual time | 30,987 | **1.24%** |
| Unusual location + unusual time | 123,608 | **0.73%** |
| Unusual time + fast transaction | 4,144 | **3.21%** |

The strongest behavioral combinations generally involve **large transaction amounts combined with unusual timing, location, or transaction velocity**.

## 📊 Behavioral Anomaly Analysis

The combined behavioral anomaly score also shows increasing fraud concentration.

| Anomaly Score | Transactions | Fraud | Fraud Rate |
|---:|---:|---:|---:|
| 0 | 192,970 | 218 | **0.113%** |
| 1 | 266,665 | 841 | **0.315%** |
| 2 | 84,449 | 749 | **0.887%** |
| 3 | 10,831 | 287 | **2.650%** |
| 4 | 839 | 49 | **5.840%** |
| 5 | 13 | 1 | **7.692%** |

This indicates that transactions exhibiting more behavioral anomalies tend to have higher fraud rates.

# 🖥️ Interactive Streamlit Dashboard

The project includes an interactive Streamlit dashboard for exploring the fraud detection system.

## Dashboard Features

### 📊 Fraud Detection KPIs

Displays:

- Total transactions
- Total fraud transactions
- APPROVE transactions
- VERIFY transactions
- BLOCK transactions
- Fraud routing recall
- Automatic BLOCK precision

### 🚦 Transaction Routing

Visualizes how transactions are distributed across:

```text
APPROVE
VERIFY
BLOCK

from pathlib import Path

content = r'''
# 📁 Project Structure

```text
hybrid-credit-card-fraud-detection/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── results/
│
├── src/
│   ├── feature_engineering.py
│   ├── train_models.py
│   ├── build_customer_profiles.py
│   ├── behavior_engine_v3_train.py
│   ├── build_behavior_train.py
│   ├── train_behavior.py
│   ├── final_hybrid.py
│   └── analyze_mistakes.py
│
├── dashboard.py
├── dashboard_data.csv
├── dashboard_metrics.json
├── FINAL_RESULTS.md
├── requirements.txt
├── .gitignore
└── README.md


