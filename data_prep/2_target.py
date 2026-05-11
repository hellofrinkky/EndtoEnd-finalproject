import pandas as pd

# =============================================================================
# 2_target.py — สร้าง Target Variable สำหรับ Supervised Learning
#
# Input : clean_cat.csv (จาก 1_clean.py)
# Output: clean_cat.csv (เพิ่ม column target_buy)
#
# [แก้ไขหลัก] ของเดิมใช้ idxmax() ได้ค่า 1-10 (10 class)
# ปัญหา: data มีแค่ ~148 rows → แต่ละ class มีน้อยมาก model train ได้แย่
#
# แก้เป็น Binary Classification แทน:
#   target_buy = 1  หากผู้ตอบ "อยากซื้อ" Option 3 ในระดับ >= 4 (เห็นด้วย/เห็นด้วยที่สุด)
#   target_buy = 0  หากให้คะแนน Option 3 ต่ำกว่านั้น
#
# เหตุผลที่เลือก Option 3:
#   - Option 3 ได้คะแนนสูงสุดจาก BA analysis (Top-3 rate 68.2%)
#   - สอดคล้องกับ Strategic Recommendation ของทีม BA
#   - Binary target ทำให้ใช้ metrics Accuracy/Precision/Recall/F1 ได้ตรงไปตรงมา
# =============================================================================

import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

df = pd.read_csv("data/processed/clean_cat.csv")

# =============================================================================
# STEP 1: สร้าง target_buy (Binary)
# threshold >= 4 หมายความว่าผู้ตอบ "เห็นด้วย" หรือ "เห็นด้วยที่สุด" กับ
# ประโยค "รู้สึกอยากซื้อสินค้า" เมื่อเห็น packaging Option 3
# =============================================================================

WINNER_OPTION = 3   # Option ที่ BA เลือกว่าดีที่สุด
THRESHOLD     = 4   # คะแนน Likert ขั้นต่ำที่ถือว่า "อยากซื้อ" (4=เห็นด้วย, 5=เห็นด้วยที่สุด)

target_col = f'opt{WINNER_OPTION}_buy'

df['target_buy'] = (df[target_col] >= THRESHOLD).astype(int)

# =============================================================================
# STEP 2: รายงาน Class Distribution
# ควรตรวจสอบว่า class ไม่ imbalanced เกินไป (ควรมีทั้งสองฝั่งพอสมควร)
# ถ้า 0 หรือ 1 มีน้อยกว่า 20% อาจต้องพิจารณาใช้ SMOTE หรือ class_weight='balanced'
# =============================================================================

print("---- Class Distribution ของ target_buy ----")
counts = df['target_buy'].value_counts().sort_index()
pct    = (counts / len(df) * 100).round(1)
print(pd.DataFrame({'count': counts, 'pct(%)': pct}))
print(f"\nจำนวนทั้งหมด: {len(df)} rows")

# แจ้งเตือนถ้า imbalanced
minority_pct = pct.min()
if minority_pct < 20:
    print(f"\n⚠️  WARNING: Class น้อยมีแค่ {minority_pct}% — แนะนำให้ ML Engineer ใช้ class_weight='balanced'")
else:
    print(f"\n✅ Class distribution โอเค (minority = {minority_pct}%)")

# =============================================================================
# STEP 3: Feature Summary — แสดง column ที่ทีม ML จะใช้เป็น Features
# =============================================================================

# แยก features ออกจาก target และ top3 flags
exclude_cols = ['target_buy'] + [f'top3_opt{i}' for i in range(1, 11)]
feature_cols = [col for col in df.columns if col not in exclude_cols]

print(f"\n---- Features สำหรับ ML ({len(feature_cols)} columns) ----")
print(feature_cols)

# =============================================================================
# STEP 4: Export
# =============================================================================

df.to_csv('data/processed/clean_cat.csv', index=False, encoding='utf-8-sig')
print("\n✅ Export data/processed/clean_cat.csv พร้อม target_buy เรียบร้อย")
print("   → ส่งต่อไฟล์นี้ให้ ML Engineer (คนที่ 3 และ 4) ได้เลย")