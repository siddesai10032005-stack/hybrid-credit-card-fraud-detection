
A two-stage machine learning system combining transaction-level fraud detection with customer behavioral verification.

## 🎯 Project Overview

The system combines two perspectives:

- **Stage 1:** Random Forest transaction-level fraud detection
- **Stage 2:** Random Forest customer behavioral verification
- **Decision Engine:** Hybrid risk scoring

Final decisions:

- 🟢 **APPROVE**
- 🟡 **VERIFY**
- 🔴 **BLOCK**

## 🏗️ System Architecture

```text
                    TRANSACTION
                         |
                         v
              +---------------------+
              |       STAGE 1       |
              |   Random Forest     |
              | Transaction Risk    |
              +----------+----------+
                         |
                  Fraud Probability
                         |
                         v
              +---------------------+
              |       STAGE 2       |
              | Behavioral Random   |
              |      Forest         |
              | Customer Behavior   |
              +----------+----------+
                         |
                Behavior Probability
                         |
                         v
              +---------------------+
              |   HYBRID RISK SCORE |
              |                     |
              | 70% Stage 1        |
              | 30% Stage 2        |
              +----------+----------+
                         |
          +--------------+--------------+
          |              |              |
       < 0.10       0.10 - <0.90       >= 0.90
          |              |              |
          v              v              v
       APPROVE        VERIFY          BLOCK




🔬 Stage 1 — Transaction-Level Fraud Detection

The first model evaluates the transaction itself and produces a fraud probability.

Features
Transaction amount
Transaction hour
Day of week
Day
Month
Weekend indicator
Customer average transaction amount
Amount ratio
Customer transaction count
Location distance
City population
Transaction category
Customer gender
Models Evaluated

Two models were evaluated:

Logistic Regression
Random Forest

Random Forest was selected as the final Stage 1 model because it performed substantially better on the fraud-detection task.

Random Forest Configuration
n_estimators = 100
max_depth = 15
class_weight = balanced
random_state = 42

The model produces:

fraud_probability
👤 Stage 2 — Customer Behavioral Verification

The second model evaluates whether a transaction is unusual compared with the customer's historical behavior.

Instead of asking only:

"Does this transaction look fraudulent?"

the behavioral model asks:

"Does this transaction look unusual for this customer?"

Behavioral Features
Amount deviation
Merchant visits
New merchant
Category visits
New category
Hour deviation
Location deviation
Seconds since previous transaction
High velocity
Log amount change
Log location change
Merchant location anomaly
Amount-location anomaly
Amount-velocity anomaly
Combined behavioral anomaly
Random Forest Configuration
n_estimators = 200
max_depth = 12
min_samples_leaf = 10
class_weight = balanced
random_state = 42

The model produces:

behavior_probability
🧮 Hybrid Risk Score

The two model outputs are combined using:

Hybrid Risk Score =
    0.70 × fraud_probability
  + 0.30 × behavior_probability

The transaction-level model contributes 70% of the final score, while customer behavioral verification contributes 30%.

🚦 Three-Way Decision System
Hybrid Score	Decision	Action
< 0.10	🟢 APPROVE	Automatically approve
0.10 - < 0.90	🟡 VERIFY	Request additional customer verification
>= 0.90	🔴 BLOCK	Automatically block

The VERIFY layer allows suspicious transactions to receive additional authentication instead of immediately blocking every transaction with elevated risk.

📊 Final Test Results

The final system was evaluated on:

Transactions       : 555,719
Fraud transactions : 2,145
Transaction Routing
Decision	Transactions
🟢 APPROVE	495,711
🟡 VERIFY	59,022
🔴 BLOCK	986
Fraud Routing
Outcome	Fraud Transactions
Fraud approved	23
Fraud verified	1,170
Fraud blocked	952
Fraud caught	2,122
Key Metrics
Overall fraud routing recall : 98.93%
Automatic BLOCK precision    : 96.55%
Metric Interpretation

Fraud routing recall — 98.93%

Percentage of fraudulent transactions that were routed to either VERIFY or BLOCK instead of being automatically approved.

Automatic BLOCK precision — 96.55%

Percentage of transactions automatically blocked that were actually fraudulent.

Important: 98.93% is overall non-approve fraud routing recall, not automatic blocking recall.

🧠 Error Analysis

The final system was analyzed to understand false positives and false negatives.

False Positives

Legitimate customers can make transactions that appear unusual.

Common examples include:

Large genuine purchase
        +
Unusual location
        +
New merchant

These transactions can receive high fraud or behavioral scores even though they are legitimate.

This is one reason the VERIFY layer is useful.

False Negatives

Some fraudulent transactions resemble the customer's normal behavior.

This demonstrates an important limitation of behavioral verification:

Fraudulent behavior can sometimes imitate legitimate customer behavior.

Therefore, behavioral analysis is treated as a complementary signal rather than a perfect fraud detector.

📈 Behavioral Insights

Several combinations of behavioral anomalies showed elevated fraud rates.

Behavioral Pattern	Transactions	Fraud Rate
Large amount + unusual location	5,619	11.03%
Large amount + new merchant	2,069	13.24%
Large amount + unusual time	5,138	22.48%
Large amount + fast transaction	279	34.05%
Unusual time + fast transaction	4,144	3.21%

These results show that combinations of behavioral signals can be more informative than individual behavioral indicators.

🔎 Risk Interpretation

The hybrid score creates a continuous risk scale.

Higher hybrid scores correspond to increasingly suspicious transactions.

The final operating policy converts that continuous score into three actions:

Low Risk
   |
   v
APPROVE
   |
   | suspicious
   v
VERIFY
   |
   | highly suspicious
   v
BLOCK

This creates a practical separation between:

Normal transactions
Transactions requiring additional authentication
High-confidence suspicious transactions
🖥️ Interactive Dashboard

The project includes an interactive Streamlit dashboard.

Dashboard Features
Final KPI metrics
APPROVE / VERIFY / BLOCK distribution
Transaction routing visualization
Risk score distribution
Fraud rate by decision
Stage 1 vs behavioral risk comparison
Transaction explorer
Risk score simulator
Run Locally
streamlit run dashboard.py

The dashboard can be deployed publicly using Streamlit Community Cloud.

📁 Project Structure
hybrid-credit-card-fraud-detection/
|
├── data/
│   ├── raw/
│   └── processed/
|
├── models/
│
├── results/
│
├── src/
│   ├── build_customer_profiles.py
│   ├── build_behavior_train.py
│   ├── behavior_engine_v3_train.py
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
⚙️ Technologies Used
Python
Pandas
NumPy
Scikit-learn
Random Forest
Logistic Regression
Matplotlib
Seaborn
Joblib
Streamlit
Plotly
Git
GitHub
🗂️ Dataset

The project uses the Sparkov synthetic credit-card transaction dataset.

The dataset contains simulated transaction information including:

Transaction timestamps
Customer/card identifiers
Merchant information
Transaction categories
Transaction amounts
Customer locations
Merchant locations
Demographic information
Fraud labels

The dataset is synthetic/simulated and does not represent real banking customers or real financial transactions.

🔐 Data Handling

Large raw datasets and trained model binaries are excluded from the GitHub repository using .gitignore.

The repository contains the source code, dashboard, analysis outputs, documentation, and lightweight dashboard data required to demonstrate the project.

This keeps the repository practical while avoiding unnecessary large files.

🛡️ Fraud Verification Concept

In a real payment system, transactions routed to VERIFY could trigger additional authentication such as:

One-time password confirmation
Banking-app confirmation
Transaction confirmation
Device authentication
Customer support verification

This project implements the scoring and routing concept but does not connect to a real banking authentication system.

⚠️ Methodological Notes and Limitations

This project is a machine-learning prototype rather than a production banking system.

1. Synthetic Dataset

The dataset is simulated and does not represent real-world banking behavior.

2. Behavioral Profiles

Customer behavioral profiles are constructed from historical training data.

The current implementation does not provide a fully strict point-in-time feature store for every training transaction.

3. Chronological Evaluation

The final test transactions are evaluated using customer profiles constructed from training history.

4. Threshold Selection

The final thresholds were selected as a project-level operating point.

A production system should tune thresholds using a dedicated validation set while preserving an untouched final holdout set.

5. Verification Is Not Fraud Confirmation

A VERIFY decision means additional authentication is required. It does not mean the transaction is confirmed as fraudulent.

6. Production Requirements

A real banking deployment would additionally require:

Real-time feature stores
Streaming transaction processing
Strict point-in-time feature generation
Device fingerprinting
Account takeover detection
Network-level fraud signals
Model monitoring
Drift detection
Human review workflows
Explainability
Security and compliance controls
🚀 Future Improvements

Potential improvements include:

Strict point-in-time behavioral features
Dedicated validation and holdout datasets
XGBoost / LightGBM
Temporal sequence models
Graph-based fraud detection
Device fingerprinting
Real-time streaming inference
Adaptive customer profiles
Cost-sensitive threshold optimization
Production monitoring
Model drift detection
Explainable AI
💡 Key Takeaway

The central idea is:

Transaction-Level Risk
          +
Customer Behavioral Risk
          |
          v
     Hybrid Risk Score
          |
          v
 APPROVE / VERIFY / BLOCK

Instead of treating every transaction as simply fraud or not fraud, the system introduces a verification layer for suspicious transactions.

On the final test set, the system routed 98.93% of fraudulent transactions away from automatic approval while achieving 96.55% precision among automatically blocked transactions.

This demonstrates how combining transaction-level machine learning with customer-specific behavioral verification can create a more practical fraud decision framework.
