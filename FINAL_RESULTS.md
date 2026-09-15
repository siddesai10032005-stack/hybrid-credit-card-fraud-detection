# Final Project Results

## Hybrid Two-Stage Credit Card Fraud Detection

### Final Architecture

Stage 1:
- Random Forest transaction-level fraud detection
- Uses transaction, temporal, customer and location features

Stage 2:
- Random Forest customer behavioral verification
- Uses amount deviation, merchant novelty, category novelty,
  time deviation, location deviation, transaction velocity
  and combined behavioral anomalies

Hybrid Score:

Hybrid Score = 70% Stage-1 Fraud Probability
             + 30% Behavioral Probability

### Final Decision Policy

| Hybrid Score | Decision |
|---:|---|
| < 0.10 | APPROVE |
| 0.10 – < 0.90 | VERIFY |
| >= 0.90 | BLOCK |

## Final Evaluation

| Metric | Result |
|---|---:|
| Total transactions | 555,719 |
| Fraud transactions | 2,145 |
| APPROVE | 495,711 |
| VERIFY | 59,022 |
| BLOCK | 986 |
| Fraud approved/missed | 23 |
| Fraud sent to VERIFY | 1,170 |
| Fraud automatically blocked | 952 |
| Fraud caught by VERIFY + BLOCK | 2,122 |
| Overall fraud routing recall | 98.93% |
| Legitimate transactions automatically blocked | 34 |
| Automatic BLOCK precision | 96.55% |

### Interpretation

98.93% fraud routing recall means that 98.93% of known fraud
transactions were either automatically blocked or routed to
customer verification.

Automatic BLOCK precision of 96.55% means that 952 of the 986
automatically blocked transactions were fraudulent.

The system intentionally uses a three-way decision instead of
forcing every transaction into a binary fraud/not-fraud decision.

VERIFY acts as a customer-behavior verification layer for
transactions that are suspicious but do not meet the threshold
for automatic blocking.

### Important Metric Definition

98.93% is NOT automatic blocking recall.

It is the recall of the combined VERIFY + BLOCK routing system.

Automatic blocking performance is represented separately by
BLOCK precision.

### Dataset

The project uses the Sparkov synthetic credit-card transaction
dataset from Kaggle. The data is simulated and should not be
described as real bank/customer transaction data.

### Methodological Notes

- The provided chronological train/test split was retained.
- Customer behavioral profiles for test transactions were built
  using training history.
- Stage-1 predictions were aligned with their original test
  transactions before behavioral sorting.
- Decision thresholds were selected for the project's operating
  point analysis.
- In a production system, thresholds should be tuned on a
  validation period and evaluated on an untouched final holdout.
- Location deviation uses coordinate-based distance rather than
  a geodesic distance calculation.
- The behavioral profile implementation is not a strict
  production-grade streaming implementation.
