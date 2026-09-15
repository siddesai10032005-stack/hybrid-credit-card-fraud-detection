python3 - <<'PY'
from pathlib import Path

content = """# 💳 Hybrid Two-Stage Credit Card Fraud Detection

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
