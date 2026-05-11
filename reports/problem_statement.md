# Problem Statement — Project Silent Salesman

## 1. Business Problem

A marketing agency managing a cat food brand needs to answer two critical questions:

1. **Who are our customers?** — The agency lacks a structured understanding of customer segments. All marketing spend is treated as a single mass-market campaign, leading to wasted budget and low conversion.
2. **What drives purchase decisions?** — The team does not know which product or packaging attributes most influence a consumer's decision to buy, making campaign messaging generic and ineffective.

**Business Goal:** Identify distinct customer personas and the key packaging/product drivers that predict purchase intent, so the marketing team can design targeted campaigns and an optimized packaging brief.

---

## 2. ML Problem Translation

| Business Question | ML Task | Technique |
|---|---|---|
| Who are our customer segments? | Clustering | K-Means (Unsupervised) |
| Are there outlier respondents distorting analysis? | Anomaly Detection | Isolation Forest (Unsupervised) |
| What structure exists in high-dimensional survey data? | Dimensionality Reduction | PCA (Unsupervised) |
| Which customers will buy Option 3? | Binary Classification | Random Forest, Logistic Regression, Gradient Boosting (Supervised) |
| What packaging features drive purchase? | Feature Importance | Random Forest Feature Importance (Supervised) |

**Target Variable:** `target_buy` — Binary (1 = respondent rates Option 3 ≥ 4/5, 0 = otherwise)

---

## 3. Dataset

- **Source:** Consumer survey on cat food packaging preferences
- **Size:** 148 respondents, ~80 features after engineering
- **Key Feature Groups:**
  - Food attributes (natural ingredients, taste, imported)
  - Packaging attributes (premium look, cat image, guarantee badge, origin symbol)
  - Demographics (age, gender, marital status)
  - Purchase intent scores for 10 packaging options

---

## 4. KPIs & Success Metrics

| Metric | Target | Achieved |
|---|---|---|
| Cluster separation (Silhouette Score) | > 0.3 | ✅ K=2 optimal |
| Anomaly detection rate | < 10% | ✅ 5.4% (8/148) |
| Best model F1-Score | > 0.75 | ✅ 0.816 (Random Forest) |
| Cross-validation stability (CV F1 std) | < 0.05 | ✅ ±0.030 |
| AUC-ROC | > 0.60 | ✅ 0.682 |

---

## 5. Expected Business Output

- **Customer Personas:** 2 actionable segments with distinct messaging strategies
- **Packaging Brief:** Data-driven design recommendations ranked by purchase impact
- **Campaign Targeting:** Channel and message recommendations per segment
- **Decision Support Dashboard:** Interactive web app for the marketing team
