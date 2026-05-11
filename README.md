# Cat Food Market Analysis (Data Science Project)

## 📌 Project Overview
โปรเจกต์นี้เป็นการวิเคราะห์ข้อมูลทางการตลาดสำหรับ **อาหารแมว** (Cat Food Market Survey) ผ่านกระบวนการ Data Science แบบ End-to-End ตั้งแต่การทำความสะอาดข้อมูล (Data Cleaning), วิเคราะห์ข้อมูลพฤติกรรมผู้บริโภคเชิงลึก (EDA & Unsupervised Learning), และสร้างโมเดลทำนายโอกาสในการซื้อสินค้า (Supervised Learning)

## 📁 Project Structure
โปรเจกต์นี้ถูกออกแบบตามมาตรฐาน **Clean Architecture** และ **Cookiecutter Data Science** เพื่อให้โค้ดเป็นระเบียบ ทำความเข้าใจง่าย และรองรับการทำ MLOps ในอนาคต

```text
cat_pkg/
│
├── frontend/              # 🖥️ สำหรับงาน Web/UI Application ในอนาคต
├── backend/               # ⚙️ สำหรับงาน API/Server ในอนาคต
│
├── data/                  # จัดเก็บข้อมูล (Data Layer)
│   ├── raw/               # ข้อมูลดิบที่ Export มาจากระบบ Survey (ห้ามแก้ไข)
│   └── processed/         # ข้อมูลที่ผ่านการทำความสะอาดแล้ว พร้อมใช้ทำ Model
│
├── data_prep/             # Data Pipeline (ทำความสะอาดและสร้าง Target Variable)
│   ├── 1_clean.py
│   └── 2_target.py
│
├── analysis/              # Unsupervised Learning & Exploratory Data Analysis (EDA)
│   └── 3_eda_unsupervised.py
│
├── models/                # Supervised Learning (Predictive Modeling)
│   └── 4_supervised.py
│
├── reports/               # รายงานผลการวิเคราะห์ในรูปแบบ Web Dashboard (HTML)
│   ├── index.html
│   ├── business_insight.html
│   ├── supervised.html
│   └── unsupervised.html
│
└── output_plots/          # กราฟและรูปภาพต่างๆ ที่สคริปต์สร้างขึ้น (Generated Assets)
```

## 🚀 How to Run the Dashboard

```bash
# Install dependencies
pip install dash dash-bootstrap-components flask pandas scikit-learn matplotlib seaborn

# Run the full app (initializes DB + starts dashboard)
python app.py
# → Open http://127.0.0.1:8050
```

The dashboard has 4 pages:
- **Home** `/` — Project overview, team, pipeline
- **Unsupervised** `/unsupervised` — K-Means, PCA, Anomaly Detection KPIs & charts
- **Supervised** `/supervised` — Model comparison, Feature Importance KPIs & charts
- **Business Insight** `/business-insight` — Segment strategy, design brief

To run the Flask API separately (port 5050):
```bash
python backend/app.py
```

---

## 🚀 How to Run the ML Pipeline
โปรเจกต์นี้แบ่ง Pipeline การทำงานออกเป็น 4 ขั้นตอนอย่างชัดเจน:

1. **Data Cleaning (`data_prep/1_clean.py`)**
   - นำเข้าไฟล์ `data/raw/CAT_FINAL.csv`
   - จัดการ Missing Value, Rename Columns, Encoding Demographics และทำ Text Classification เล็กน้อย
   - ส่งออกข้อมูลไปที่ `data/processed/clean_cat.csv`

2. **Target Engineering (`data_prep/2_target.py`)**
   - สร้าง Label/Target variable `target_buy` (Binary Classification) ว่าผู้บริโภคจะซื้อแพ็กเกจที่ชนะ (Option 3) หรือไม่

3. **Unsupervised Learning & EDA (`analysis/3_eda_unsupervised.py`)**
   - ทำ Descriptive Statistics, Correlation Analysis
   - ใช้ **K-Means Clustering** แบ่งกลุ่ม Persona ลูกค้า (พร้อมวิเคราะห์ Customer Persona)
   - ใช้ **Isolation Forest** ทำ Anomaly Detection

4. **Supervised Learning (`models/4_supervised.py`)**
   - เทรนโมเดล Classification (Logistic Regression, Random Forest, Gradient Boosting)
   - เปรียบเทียบประสิทธิภาพโมเดล และดึง **Feature Importance** ว่าปัจจัยใดมีผลต่อการซื้อมากที่สุด

## 🛠 Tech Stack & Dependencies
- **Language:** Python 3.x
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn
- **Visualization:** Matplotlib, Seaborn

## 📈 Business Value
ผลลัพธ์จากโปรเจกต์นี้ช่วยให้ทีม Business/Marketing เข้าใจ:
- **Persona ของกลุ่มลูกค้าเป้าหมาย:** ใครคือผู้ซื้อหลัก และให้คุณค่ากับอะไร (เช่น รสชาติ, วัตถุดิบธรรมชาติ)
- **Key Drivers:** ปัจจัยใดบนบรรจุภัณฑ์ (Packaging) หรือคุณสมบัติผลิตภัณฑ์ (Product Attributes) ที่ส่งผลต่อการ "ตัดสินใจซื้อ" อย่างมีนัยสำคัญ
