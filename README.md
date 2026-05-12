<div align="center">

# 🐱 Silent Salesman

### Cat Food Market Analysis — Data-Driven Packaging & Marketing Strategy

*เปลี่ยน Survey 148 คน ให้กลายเป็น Packaging Brief และ Marketing Strategy ที่ขับเคลื่อนด้วยข้อมูล*

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.1.0-000000?logo=flask)](https://flask.palletsprojects.com)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Vercel](https://img.shields.io/badge/Vercel-Ready-000000?logo=vercel)](https://vercel.com)

</div>

---

## 📌 Problem Statement

Marketing agency ที่ดูแลแบรนด์อาหารแมวเผชิญกับ 2 ปัญหาหลัก:

1. **ไม่รู้จักลูกค้า** — ใช้งบการตลาดแบบ Mass Market ทำให้เสียงบโดยเปล่าประโยชน์
2. **ไม่รู้ว่าอะไรขาย** — ไม่ทราบว่า Feature ไหนบนบรรจุภัณฑ์ที่ผลักดันการตัดสินใจซื้อจริงๆ

**เป้าหมาย:** ระบุ Customer Persona ที่ชัดเจน และ Key Packaging Drivers เพื่อออกแบบแคมเปญและบรรจุภัณฑ์ที่ตรงเป้า

---

## 🏆 Results at a Glance

<table>
<tr>
<th>Metric</th><th>Target</th><th>Achieved</th>
</tr>
<tr><td>Best Model F1-Score</td><td>> 0.75</td><td>✅ <strong>0.816</strong> (Random Forest)</td></tr>
<tr><td>Cross-Validation F1 (std)</td><td>< 0.05</td><td>✅ 0.799 ± 0.030</td></tr>
<tr><td>AUC-ROC</td><td>> 0.60</td><td>✅ 0.682</td></tr>
<tr><td>Anomaly Rate</td><td>< 10%</td><td>✅ 5.4% (8/148 คน)</td></tr>
<tr><td>Cluster Separation (Silhouette)</td><td>> 0.3</td><td>✅ K=2 optimal</td></tr>
</table>

**🎯 Top 3 Packaging Features ที่ขับเคลื่อนการซื้อ:**

| Rank | Feature | Description | Importance |
|:---:|---|---|:---:|
| 🥇 | `packaging_13` | ภาพแมวบนบรรจุภัณฑ์ | **10.7%** |
| 🥈 | `packaging_19` | ความดูพรีเมียม | **7.4%** |
| 🥉 | `packaging_17` | Badge รับรองคุณภาพ | **6.7%** |

---

## 🧠 ML Pipeline

```
RAW DATA (148 respondents, ~80 features)
        │
        ▼
┌─────────────────────┐
│  1_clean.py         │  Missing values, rename, encode demographics
│  2_target.py        │  สร้าง target_buy (Binary: score ≥ 4/5)
└────────┬────────────┘
         │
    ┌────┴────┐
    ▼         ▼
UNSUPERVISED  SUPERVISED
    │              │
    ├─ PCA         ├─ Logistic Regression
    ├─ K-Means     ├─ Random Forest ⭐
    └─ Iso Forest  └─ Gradient Boosting
         │              │
         └──────┬───────┘
                ▼
        DASHBOARD + API
        (Flask + SQLite)
```

### Step 1 — Data Cleaning (`data_prep/1_clean.py`)
- จัดการ Missing Values, Rename Columns, Encode Demographics
- Output: `data/processed/clean_cat.csv`

### Step 2 — Target Engineering (`data_prep/2_target.py`)
- สร้าง `target_buy` (Binary): 1 ถ้าผู้ตอบให้คะแนน Option 3 ≥ 4/5
- เป็น Proxy สำหรับ "Purchase Intent" ของ Winning Package

### Step 3 — Unsupervised Learning (`analysis/3_eda_unsupervised.py`)
- **PCA** — ลด Dimensionality เพื่อ Visualize โครงสร้างข้อมูล
- **K-Means (K=2)** — แบ่ง Customer Persona 2 กลุ่ม (Silhouette optimal)
- **Isolation Forest** — ตรวจจับ Outlier พบ 5.4% (8/148 คน)

### Step 4 — Supervised Learning (`models/4_supervised.py`)
- Train 3 models ด้วย 5-Fold Cross-Validation
- Evaluate ด้วย Confusion Matrix, ROC Curve, Feature Importance
- **Winner: Random Forest** — F1=0.816, CV F1=0.799±0.030

---

## 📊 Model Comparison

| Model | CV F1 | ± std | Accuracy | Precision | Recall | **F1** | AUC-ROC |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest** ⭐ | **0.799** | 0.030 | 0.700 | 0.741 | **0.909** | **0.816** | **0.682** |
| Gradient Boosting | 0.775 | 0.027 | 0.667 | 0.714 | 0.909 | 0.800 | 0.562 |
| Logistic Regression | 0.705 | 0.039 | 0.600 | 0.750 | 0.682 | 0.714 | 0.580 |

> Random Forest ชนะทั้ง F1 และ AUC-ROC พร้อม CV std ต่ำสุด — แสดงถึงความ stable ที่สุดในทุก fold

---

## 👥 Customer Personas

### 🟦 Cluster 0 — Value-Driven & Functionality Seekers

> *"ซื้อเพราะมันดีต่อสุขภาพแมว ไม่ใช่เพราะมันดูแพง"*

ให้ความสำคัญกับ **ประโยชน์ใช้สอย** และความคุ้มค่า เลือกซื้อจากสรรพคุณที่ชัดเจน เช่น ลดก้อนขน, ดูแลสุขภาพไต

| | |
|---|---|
| **Channel** | Pet Shop, คลินิกสัตว์ |
| **Message** | สุขภาพแมว + ความคุ้มค่า |
| **Promotion** | Volume Discount, แถมของเล่น |

---

### 🟨 Cluster 1 — Premium & Aesthetic Lovers

> *"แมวคือสมาชิกในครอบครัว สิ่งที่ดีที่สุดเท่านั้น"*

มองแมวเหมือนสมาชิกในครอบครัว ยินดีจ่ายแพงกว่า ให้ความสำคัญกับ **แบรนด์ที่ดูพรีเมียม** และ Badge รับรอง

| | |
|---|---|
| **Channel** | Supermarket Hi-End, Official Online Store |
| **Message** | "สิ่งที่ดีที่สุดสำหรับลูกรัก" + Human-grade ingredients |
| **Design** | โทนสีดำ-ทอง หรือ Minimalist + Badge รับรองระดับสากล |

> 📌 **Strategic Recommendation:** Key Drivers จาก Feature Importance (ภาพแมว, ความพรีเมียม, Badge) สอดคล้องกับ Cluster 1 อย่างชัดเจน — แนะนำให้ Position สินค้าใหม่ใน **Premium Segment** เป็นหลัก

---

## 📁 Project Structure

```
cat_pkg/
│
├── app.py                          # 🚀 Entry point
│
├── frontend/                       # 🖥️ HTML Dashboard (4 หน้า)
│   ├── index.html                  # Home — overview & pipeline
│   ├── unsupervised.html           # K-Means, PCA, Anomaly KPIs
│   ├── supervised.html             # Model comparison, ROC, Feature Importance
│   └── business_insight.html       # Segment strategy & Packaging brief
│
├── backend/
│   ├── app.py                      # Flask routes & REST API
│   └── db.py                       # SQLite initialization
│
├── data/
│   ├── raw/CAT_FINAL.csv           # ⚠️ ข้อมูลดิบ — ห้ามแก้ไข
│   └── processed/clean_cat.csv     # ข้อมูลที่ผ่าน cleaning แล้ว
│
├── data_prep/
│   ├── 1_clean.py                  # Data Cleaning & Encoding
│   └── 2_target.py                 # Target Variable Engineering
│
├── analysis/
│   └── 3_eda_unsupervised.py       # EDA, K-Means, PCA, Isolation Forest
│
├── models/
│   └── 4_supervised.py             # Classification & Feature Importance
│
├── output_plots/                   # Auto-generated charts & CSVs
│   └── supervised/
│
├── reports/                        # Business documents
│   ├── problem_statement.md
│   ├── strategic_recommendations.md
│   ├── design_brief.md
│   ├── machine_learning_methodology.md
│   └── system_architecture.md
│
├── requirements.txt
└── vercel.json                     # Vercel deployment config
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the ML Pipeline

> รันครั้งแรก หรือเมื่อข้อมูลเปลี่ยน — ผลลัพธ์จะถูก save ไว้ใน `output_plots/`

```bash
python data_prep/1_clean.py
python data_prep/2_target.py
python analysis/3_eda_unsupervised.py
python models/4_supervised.py
```

### 3. Start the Dashboard

```bash
python app.py
```

เปิด browser ที่ **http://127.0.0.1:8050**

---

## 🔌 REST API Reference

Base URL: `http://127.0.0.1:8050`

| Method | Endpoint | Description |
|:---:|---|---|
| `GET` | `/api/cluster-profile` | K-Means cluster statistics |
| `GET` | `/api/model-comparison` | Accuracy, F1, AUC-ROC per model |
| `GET` | `/api/feature-importance` | Top features ranked by importance |
| `GET` | `/api/descriptive-stats` | Descriptive statistics of survey data |
| `GET` | `/api/persona` | Demographic breakdown per cluster |
| `GET` | `/api/anomaly-summary` | Anomaly detection summary |

ทุก endpoint คืนค่าเป็น JSON array

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.x |
| Web Framework | Flask 3.1.0 |
| Data Processing | Pandas 2.2.3 |
| Machine Learning | Scikit-Learn |
| Visualization | Matplotlib, Seaborn |
| Frontend | HTML / CSS / Vanilla JavaScript |
| Database | SQLite |
| Deployment | Vercel (`vercel.json`) |

---

## 📄 Reports

| ไฟล์ | เนื้อหา |
|---|---|
| `reports/problem_statement.md` | Business problem, ML translation, KPIs |
| `reports/strategic_recommendations.md` | Marketing strategy per segment |
| `reports/design_brief.md` | Packaging design brief จาก Feature Importance |
| `reports/machine_learning_methodology.md` | ML methodology อย่างละเอียด |
| `reports/system_architecture.md` | System architecture diagram |
