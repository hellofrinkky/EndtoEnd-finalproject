import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

# Setup font สำหรับภาษาไทยใน matplotlib (ป้องกันปัญหาฟอนต์สี่เหลี่ยม □□□)
try:
    thai_fonts = [f for f in fm.findSystemFonts() if any(n in f.lower() for n in ['thsarabun', 'noto', 'tahoma'])]
    if thai_fonts:
        plt.rcParams['font.family'] = fm.FontProperties(fname=thai_fonts[0]).get_name()
except Exception:
    pass

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report,
                              roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# =============================================================================
# 4_supervised.py — Supervised Learning (Binary Classification)
#
# Input : clean_cat.csv (จาก 3_eda_unsupervised.py — มี cluster label แล้ว)
# Output:
#   - model_comparison.csv      → เปรียบเทียบ metrics ทุกโมเดล
#   - feature_importance.csv    → feature importance จาก best model
#   - รูปใน output_plots/supervised/
#
# Target: target_buy (1 = อยากซื้อ Option 3, 0 = ไม่อยาก)
#
# Models:
#   1. Logistic Regression  — Baseline, interpretable
#   2. Random Forest        — Handle mixed features, robust
#   3. Gradient Boosting    — มักให้ accuracy สูง (แทน XGBoost เพื่อไม่ต้องติดตั้งเพิ่ม)
# =============================================================================

os.makedirs('output_plots/supervised', exist_ok=True)

# =============================================================================
# LOAD DATA
# =============================================================================

df = pd.read_csv('data/processed/clean_cat.csv')
print(f"✅ โหลดข้อมูล: {df.shape[0]} rows, {df.shape[1]} columns")

# =============================================================================
# SECTION 1: FEATURE SELECTION
# เลือก features ที่เหมาะสมสำหรับ predict purchase intent
# แยก 3 กลุ่ม:
#   demographics   — อายุ, เพศ, สถานภาพ
#   preferences    — food/packaging attributes (ความสำคัญต่อการตัดสินใจ)
#   cluster        — segment ที่ได้จาก unsupervised (เพิ่ม context)
#
# ไม่ใช้ option scores ของ Option อื่น (opt1_buy, opt2_buy ฯลฯ)
# เพราะจะทำให้ model leaky — รู้คำตอบล่วงหน้า
# =============================================================================

demographics = ['age_', 'gender', 'status', 'mean_2', 'packaging_10', 'packaging_11']

food_cols = ['food_natural_5', 'food_import_6', 'food_taste_7',
             'food_make_import_8', 'brand_popular_9']

pkg_cols  = ['packaging_12', 'packaging_13', 'packaging_14', 'packaging_15',
             'packaging_16', 'packaging_17', 'packaging_18', 'packaging_19']

cat_flags = ['cat_thai', 'cat_inter', 'cat_mixed',
             'price_premium', 'price_medium', 'price_low',
             'pkg_none', 'pkg_quality', 'pkg_packaging', 'pkg_gift', 'pkg_other']

# [แก้ไข] นำ 'cluster' ออกเพื่อป้องกัน Data Leakage ตามคำแนะนำ
# เนื่องจาก cluster ถูกสร้างมาจากข้อมูลทั้งหมด (train+test) ในขั้นตอน Unsupervised
# การนำมาใช้เป็น feature จะทำให้โมเดลแอบเห็นข้อมูล test ล่วงหน้า

# ตรวจสอบว่า columns มีอยู่จริง (ป้องกัน KeyError)
all_features = demographics + food_cols + pkg_cols + cat_flags
all_features = [c for c in all_features if c in df.columns]

TARGET = 'target_buy'
assert TARGET in df.columns, f"❌ ไม่พบ column '{TARGET}' — รัน 2_target.py ก่อน"

X = df[all_features].copy()
y = df[TARGET].copy()

print(f"\nFeatures: {len(all_features)} columns")
print(f"Target distribution:")
print(y.value_counts().to_string())
print(f"  Class balance: {y.mean():.1%} positive")

# =============================================================================
# SECTION 2: TRAIN/TEST SPLIT
# ใช้ stratify=y เพื่อให้ class ratio เท่ากันใน train/test
# test_size=0.2 → 80% train, 20% test
# =============================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")
print(f"Train positive rate: {y_train.mean():.1%}")
print(f"Test  positive rate: {y_test.mean():.1%}")

# =============================================================================
# SECTION 3: DEFINE MODELS
# ใช้ Pipeline (Scaler + Model) เพื่อป้องกัน data leakage
# StandardScaler จะ fit เฉพาะ train set แล้วค่อย transform test set
# =============================================================================

# class_weight='balanced' ช่วยจัดการ imbalanced class โดยอัตโนมัติ
models = {
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        ))
    ]),
    'Random Forest': Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier(
            n_estimators=200,
            class_weight='balanced',
            max_depth=6,
            min_samples_leaf=3,
            random_state=42
        ))
    ]),
    'Gradient Boosting': Pipeline([
        ('scaler', StandardScaler()),
        ('model', GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        ))
    ]),
}

# =============================================================================
# SECTION 4: TRAIN + EVALUATE
# ประเมินด้วย 5-Fold Stratified Cross-Validation บน train set
# และ holdout test set
# =============================================================================

print("\n" + "="*60)
print("SECTION 4: MODEL TRAINING & EVALUATION")
print("="*60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, pipeline in models.items():
    print(f"\n▶ {name}")

    # Cross-Validation บน train set (F1 เพราะ data อาจ imbalanced)
    cv_f1 = cross_val_score(pipeline, X_train, y_train, cv=cv,
                             scoring='f1', n_jobs=-1)
    cv_acc = cross_val_score(pipeline, X_train, y_train, cv=cv,
                              scoring='accuracy', n_jobs=-1)

    print(f"  CV F1:       {cv_f1.mean():.3f} ± {cv_f1.std():.3f}")
    print(f"  CV Accuracy: {cv_acc.mean():.3f} ± {cv_acc.std():.3f}")

    # Train บน full train set แล้ว predict test set
    pipeline.fit(X_train, y_train)
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_proba)

    print(f"  Test Accuracy:  {acc:.3f}")
    print(f"  Test Precision: {prec:.3f}")
    print(f"  Test Recall:    {rec:.3f}")
    print(f"  Test F1:        {f1:.3f}")
    print(f"  Test AUC-ROC:   {auc:.3f}")

    results[name] = {
        'CV_F1_mean': round(cv_f1.mean(), 3),
        'CV_F1_std':  round(cv_f1.std(), 3),
        'CV_Acc_mean': round(cv_acc.mean(), 3),
        'Accuracy':  round(acc, 3),
        'Precision': round(prec, 3),
        'Recall':    round(rec, 3),
        'F1':        round(f1, 3),
        'AUC_ROC':   round(auc, 3),
        'pipeline':  pipeline,
        'y_pred':    y_pred,
        'y_proba':   y_proba,
    }

# =============================================================================
# SECTION 5: MODEL COMPARISON TABLE
# =============================================================================

print("\n" + "="*60)
print("SECTION 5: MODEL COMPARISON")
print("="*60)

comparison_cols = ['CV_F1_mean', 'CV_F1_std', 'CV_Acc_mean',
                   'Accuracy', 'Precision', 'Recall', 'F1', 'AUC_ROC']
comparison_df = pd.DataFrame({
    name: {k: v for k, v in r.items() if k in comparison_cols}
    for name, r in results.items()
}).T

print(comparison_df.to_string())
comparison_df.to_csv('output_plots/supervised/model_comparison.csv')
print("\n✅ บันทึก model_comparison.csv")

# Plot comparison bar chart
fig, axes = plt.subplots(1, 4, figsize=(16, 5))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
colors  = ['#3498db', '#e74c3c', '#2ecc71']

for ax, metric in zip(axes, metrics):
    vals = comparison_df[metric].values
    bars = ax.bar(comparison_df.index, vals, color=colors)
    ax.set_title(metric, fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.set_xticklabels(comparison_df.index, rotation=15, ha='right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

plt.suptitle('Model Performance Comparison', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('output_plots/supervised/model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ บันทึก model_comparison.png")

# =============================================================================
# SECTION 6: SELECT BEST MODEL
# เลือกจาก F1 score (เหมาะกับ imbalanced data มากกว่า accuracy)
# =============================================================================

best_name = comparison_df['F1'].idxmax()
best      = results[best_name]
print(f"\n🏆 Best Model: {best_name} (F1 = {best['F1']:.3f})")

# =============================================================================
# SECTION 7: CONFUSION MATRIX (Best Model)
# =============================================================================

cm = confusion_matrix(y_test, best['y_pred'])
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['ไม่ซื้อ (0)', 'อยากซื้อ (1)'],
            yticklabels=['ไม่ซื้อ (0)', 'อยากซื้อ (1)'])
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('Actual', fontsize=12)
ax.set_title(f'Confusion Matrix — {best_name}', fontsize=13)
plt.tight_layout()
plt.savefig('output_plots/supervised/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ บันทึก confusion_matrix.png")

print(f"\nClassification Report ({best_name}):")
print(classification_report(y_test, best['y_pred'],
                             target_names=['ไม่ซื้อ', 'อยากซื้อ']))

# =============================================================================
# SECTION 8: ROC CURVE (ทุก model)
# =============================================================================

fig, ax = plt.subplots(figsize=(7, 6))
colors_roc = ['#3498db', '#e74c3c', '#2ecc71']

for (name, r), color in zip(results.items(), colors_roc):
    fpr, tpr, _ = roc_curve(y_test, r['y_proba'])
    ax.plot(fpr, tpr, color=color, linewidth=2,
            label=f"{name} (AUC={r['AUC_ROC']:.3f})")

ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Baseline')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curve Comparison', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('output_plots/supervised/roc_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ บันทึก roc_curve.png")

# =============================================================================
# SECTION 9: FEATURE IMPORTANCE (Best Model)
# Random Forest / Gradient Boosting → feature_importances_
# Logistic Regression → |coef_|
# =============================================================================

print("\n" + "="*60)
print("SECTION 9: FEATURE IMPORTANCE")
print("="*60)

best_pipeline = best['pipeline']
best_model    = best_pipeline.named_steps['model']

if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
elif hasattr(best_model, 'coef_'):
    importances = np.abs(best_model.coef_[0])
else:
    importances = np.ones(len(all_features))

feat_imp = pd.DataFrame({
    'feature':    all_features,
    'importance': importances
}).sort_values('importance', ascending=False).reset_index(drop=True)

print(feat_imp.head(10).to_string(index=False))
feat_imp.to_csv('output_plots/supervised/feature_importance.csv', index=False)
print("\n✅ บันทึก feature_importance.csv")

# Plot top 15 features
top_n = min(15, len(feat_imp))
fig, ax = plt.subplots(figsize=(9, 6))
colors_bar = ['#e74c3c' if i < 3 else '#3498db' for i in range(top_n)]
ax.barh(feat_imp['feature'][:top_n][::-1],
        feat_imp['importance'][:top_n][::-1],
        color=colors_bar[::-1])
ax.set_xlabel('Importance Score', fontsize=11)
ax.set_title(f'Feature Importance — {best_name}\n(Top {top_n} features, red = top 3)',
             fontsize=12)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('output_plots/supervised/feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ บันทึก feature_importance.png")

# =============================================================================
# SECTION 10: MODEL JUSTIFICATION SUMMARY
# อธิบายว่าทำไมถึงเลือก model นี้ เพื่อใช้ในรายงาน
# =============================================================================

print("\n" + "="*60)
print("SECTION 10: MODEL JUSTIFICATION")
print("="*60)

justification = f"""
🏆 Best Model: {best_name}

เหตุผลที่เลือก:
  - F1 Score สูงสุด = {best['F1']:.3f}
    (F1 เหมาะกับ binary classification ที่ data อาจ imbalanced)
  - AUC-ROC = {best['AUC_ROC']:.3f}
    (ยิ่งใกล้ 1 ยิ่งดี — สะท้อนความสามารถแยก class ได้จริง)
  - CV F1 = {best['CV_F1_mean']:.3f} ± {best['CV_F1_std']:.3f}
    (cross-validation ยืนยันว่าไม่ overfit)

ข้อจำกัด:
  - Dataset มีเพียง {len(df)} rows — ผลอาจมี variance สูง
  - แนะนำ collect data เพิ่มในอนาคตเพื่อ model ที่ robust กว่านี้

Top 3 features ที่ส่งผลต่อการซื้อ Option 3 มากที่สุด:
"""

for i, row in feat_imp.head(3).iterrows():
    justification += f"  {i+1}. {row['feature']} (importance={row['importance']:.4f})\n"

print(justification)

with open('output_plots/supervised/model_justification.txt', 'w', encoding='utf-8') as f:
    f.write(justification)
print("✅ บันทึก model_justification.txt")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "="*60)
print("Output files ใน output_plots/supervised/:")
for fname in sorted(os.listdir('output_plots/supervised')):
    print(f"  - {fname}")
print("="*60)
print("\n✅ 4_supervised.py เสร็จสมบูรณ์")
print("   → พร้อม handoff ให้ Frontend Developer (คนที่ 5) นำ plots ไปใช้ใน Dashboard")