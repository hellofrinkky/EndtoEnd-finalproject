import pandas as pd

# =============================================================================
# [DATA SOURCE]
# CAT_FINAL.csv = ข้อมูล raw survey Case 5 (อาหารแมว) ที่ export มาจาก Excel Sheet "5"
# กรอง + แปลงข้อมูลด้วยไฟล์นี้ → ได้ clean_cat.csv สำหรับทีม ML
# =============================================================================

import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

df = pd.read_csv('data/raw/CAT_FINAL.csv')

# ลบ Timestamp เพราะไม่ได้ใช้ใน ML
df = df.drop(columns=['Timestamp'], errors='ignore')

print(df.columns[:5])

# =============================================================================
# SECTION 1: RENAME COLUMNS
# ตั้งชื่อ column ย่อ + suffix เลขคำถาม เพื่อ track กลับ survey ได้ง่าย
#
# [แก้ไข] ใช้ regex-based rename แทน dict เพราะ column จริงใน Excel
# มีจำนวนช่องว่างไม่สม่ำเสมอก่อน "[" (บาง option มี 1 เว้น บาง option มี 2 เว้น)
# การ rename ด้วย dict ธรรมดาจึง match ไม่ได้ → ใช้ str.strip() + keyword แทน
# =============================================================================

import re

# Step 1: Rename column ที่ชื่อชัดเจน (ไม่ใช่ Option) ด้วย dict ปกติ
# ใช้ strip() normalize whitespace ทั้งหมดก่อน
df.columns = df.columns.str.strip()

non_option_rename = {
    'คุณเคยซื้ออาหารแมวและมีประสบการณ์เลี้ยงแมวในบ้านหรือไม่': 'had_none_1',
    'อายุของคุณ': 'age_',
    'เพศของคุณ': 'gender',
    'สถานภาพสมรส': 'status',
    'แมวมีความหมายอย่างไรต่อคุณ เช่น เป็นเหมือนลูกของเรา, เป็นเพื่อนคลายเหงา [พิมพ์ตามความรู้สึกได้เลย]': 'cat_mean_2',
    'คุณเลี้ยงแมวพันธุ์ใด [โปรดพิมพ์ระบุ]': 'cat_kind_3',
    'ปัจจุบันคุณซื้ออาหารแมวสำเร็จรูปชนิดเม็ดแบรนด์ใด[โปรดพิมพ์ระบุ]': 'brand_4',
    'คุณสมบัติของอาหารแมวสำเร็จรูปชนิดเม็ดที่ส่งผลต่อการตัดสินใจซื้อ [ใช้วัตถุดิบจากธรรมชาติ]': 'food_natural_5',
    'คุณสมบัติของอาหารแมวสำเร็จรูปชนิดเม็ดที่ส่งผลต่อการตัดสินใจซื้อ [ใช้วัตถุดิบนำเข้าจากต่างประเทศ เช่น เนื้อปลาทูน่าจากญี่ปุ่น]': 'food_import_6',
    'คุณสมบัติของอาหารแมวสำเร็จรูปชนิดเม็ดที่ส่งผลต่อการตัดสินใจซื้อ [รสชาติกลมกล่อมอร่อยถูกปากแมว เช่น เทไว้แล้วแมวกินหมดไม่เหลือ, หยิบถุงแล้วแมวรอกิน]': 'food_taste_7',
    'คุณสมบัติของอาหารแมวสำเร็จรูปชนิดเม็ดที่ส่งผลต่อการตัดสินใจซื้อ [เป็นผลิตภัณฑ์จากต่างประเทศ เช่น ญี่ปุ่น, อเมริกา]': 'food_make_import_8',
    'คุณสมบัติของอาหารแมวสำเร็จรูปชนิดเม็ดที่ส่งผลต่อการตัดสินใจซื้อ [แบรนด์มีชื่อเสียงเป็นที่รู้จัก]': 'brand_popular_9',
    'บรรจุภัณฑ์ (packaging) มีผลต่อการตัดสินใจซื้อใจหรือไม่': 'packaging_10',
    'สำหรับบรรจุภัณฑ์อาหารแมว คุณชอบภาพแบบใด': 'packaging_11',
    'บรรจุภัณฑ์ของอาหารแมวสำเร็จรูปชนิดเม็ดที่ส่งผลต่อการตัดสินใจซื้อ [บรรจุภัณฑ์ดูดีพรีเมียม]': 'packaging_12',
    'บรรจุภัณฑ์ของอาหารแมวสำเร็จรูปชนิดเม็ดที่ส่งผลต่อการตัดสินใจซื้อ [บรรจุภัณฑ์มีภาพแมว]': 'packaging_13',
    'บรรจุภัณฑ์ของอาหารแมวสำเร็จรูปชนิดเม็ดที่ส่งผลต่อการตัดสินใจซื้อ [บรรจุภัณฑ์มีภาพอาหารเม็ด รูปทรงของอาหารเม็ดจริงให้เห็น]': 'packaging_14',
    'บรรจุภัณฑ์ของอาหารแมวสำเร็จรูปชนิดเม็ดที่ส่งผลต่อการตัดสินใจซื้อ [บรรจุภัณฑ์มีภาพวัตถุดิบและส่วนผสมจริงให้เห็น]': 'packaging_15',
    'บรรจุภัณฑ์ของอาหารแมวสำเร็จรูปชนิดเม็ดที่ส่งผลต่อการตัดสินใจซื้อ [บรรจุภัณฑ์เป็นมิตรต่อสิ่งแวดล้อม]': 'packaging_16',
    'บรรจุภัณฑ์ของอาหารแมวสำเร็จรูปชนิดเม็ดที่ส่งผลต่อการตัดสินใจซื้อ [มีสัญลักษณ์สื่อถึงแหล่งผลิตหรือที่มา เช่น นำเข้าจากประเทศx]': 'packaging_17',
    'บรรจุภัณฑ์ของอาหารแมวสำเร็จรูปชนิดเม็ดที่ส่งผลต่อการตัดสินใจซื้อ [มีสัญลักษณ์์สื่อถึงประโยชน์หรือฟังก์ชั่น เช่น ช่วยลดก้อนขน]': 'packaging_18',
    'บรรจุภัณฑ์ของอาหารแมวสำเร็จรูปชนิดเม็ดที่ส่งผลต่อการตัดสินใจซื้อ [มีการการันตี เช่น ได้รับรางวัล, ยอดขายอันดับ 1]': 'packaging_19',
    'ถ้าคุณสามารถเพิ่มอะไรก็ได้เข้าไปในบรรจุภัณฑ์ของอาหารแมว คุณอยากเติมอะไร [พิมพ์ตามใจได้เลย]': 'add_pkg_20',
    'จากตัวเลือกทั้งหมด คุณชอบการออกแบบบรรจุภัณฑ์อาหารแมวสำเร็จรูปแบบใดมากที่สุด 3 อันดับแรก': 'top_3'
}
df.rename(columns=non_option_rename, inplace=True)

# Step 2: Rename Option columns ด้วย regex
# [แก้ไข BUG] ปัญหาเดิม: ชื่อ column ใน Excel มีช่องว่างไม่สม่ำเสมอก่อน "["
# Option 1-7 มี 2 ช่องว่าง, Option 8-10 มี 1 ช่องว่าง → dict ธรรมดา match ไม่ได้
# แก้โดย normalize ช่องว่างใน column name ก่อน (แปลง 2+ ช่องว่าง → 1 ช่องว่าง)
# แล้วค่อย map ด้วย keyword ที่รู้จัก

# Normalize whitespace ใน column names ทั้งหมด (2+ spaces → 1 space)
df.columns = [re.sub(r' {2,}', ' ', c) for c in df.columns]

# map keyword ใน bracket → suffix ชื่อ column
# [แก้ไข] ใช้ substring matching แทน exact string เพราะ Unicode ของคำว่า "รูู้"
# ใน Excel มี codepoint ต่างลำดับกับที่พิมพ์ใน source code → exact match ล้มเหลว
# วิธีนี้ match จาก keyword สั้นๆ ที่ไม่มีปัญหา Unicode แทน

def get_suffix(keyword):
    """ดึง suffix จาก keyword ใน bracket โดยใช้ substring ที่ปลอดภัย"""
    if 'อยากซื้อ' in keyword:
        return 'buy'
    if 'โดดเด่น' in keyword or 'แตกต่าง' in keyword:
        return 'unique'
    if 'พรีเมียม' in keyword:
        return 'premium'
    if 'รสชาติ' in keyword or 'กลมกล่อม' in keyword:
        return 'taste'
    if 'เหมาะกับตัวฉัน' in keyword:
        return 'personal'
    return None

def rename_option_col(col):
    """
    แปลง 'Option N คุณคิดเห็น... [keyword]' → 'optN_suffix'
    ถ้าไม่ใช่ Option column ให้ return ชื่อเดิม
    """
    m = re.match(r'Option (\d+) .+\[(.+)\]', col)
    if not m:
        return col
    n, keyword = m.group(1), m.group(2).strip()
    suffix = get_suffix(keyword)
    if suffix:
        return f'opt{n}_{suffix}'
    return col  # ถ้า keyword ไม่รู้จัก ให้ return เดิม

df.columns = [rename_option_col(c) for c in df.columns]

# ตรวจสอบว่า rename สำเร็จ — ควรเจอ opt1_buy ถึง opt10_personal
opt_cols_found = [c for c in df.columns if c.startswith('opt') and '_' in c]
print(f"Option columns ที่ rename สำเร็จ: {len(opt_cols_found)} columns")
if len(opt_cols_found) != 50:
    print("⚠️  WARNING: ควรได้ 50 columns (Option 1-10 × 5 ด้าน) — เช็ค column ที่เหลือ:")
    print([c for c in df.columns if 'Option' in c])

# =============================================================================
# SECTION 2: FILTER — เก็บเฉพาะคนที่เคยซื้ออาหารแมว
# คนที่ตอบ "ไม่เคย" จะมี column ส่วนใหญ่ว่าง ไม่สามารถใช้ใน ML ได้
# =============================================================================

change_1 = {'เคย': 1, 'ไม่เคย': 0}
df['num_1'] = df['had_none_1'].map(change_1)

print("---- เคย/ไม่เคย(เลี้ยงแมว/ซื้ออาหารแมว) ----")
print(df['num_1'].value_counts().sort_index())

df = df[df['num_1'] == 1]
df = df.reset_index(drop=True)

print("---- ลบคนไม่เคย ----")
print(df['had_none_1'].value_counts())

# =============================================================================
# SECTION 3: MISSING VALUE REPORT
# [แก้ไข] เพิ่ม missing value report ก่อน fillna เพื่อ document ใน Data Preprocessing Report
# ทีมต้องรายงานตัวเลขนี้ว่าแต่ละ column หายไปกี่ % ก่อนที่จะ fill
# =============================================================================

print("\n---- Missing Value Report (หลังกรองแล้ว) ----")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_report = pd.DataFrame({'missing_count': missing, 'missing_pct': missing_pct})

# ปริ้นท์สรุปออกทางหน้าจอ
print(missing_report.to_string())

# [แก้ไข] บันทึก Missing Value Report เป็นไฟล์อย่างชัดเจนตาม Requirement
os.makedirs('reports', exist_ok=True)
missing_report.to_csv('reports/missing_value_report.csv')
print("✅ บันทึก reports/missing_value_report.csv")

# =============================================================================
# SECTION 4: ENCODE DEMOGRAPHICS
# =============================================================================

# อายุ: แปลงช่วงอายุเป็นตัวเลข ordinal (1=น้อยสุด, 4=มากสุด)
age_x = {
    "20-29ปี": 1,
    "30-39ปี": 2,
    "40-49ปี": 3,
    "50ปี ขึ้นไป": 4
}
df["age_"] = df["age_"].map(age_x)
print("\n---- สรุปจำนวนผู้ตอบตามช่วงอายุ ----")
print(df["age_"].value_counts().sort_index())

# เพศ: แปลงเป็นตัวเลข (ชาย=3, หญิง=2, อื่นๆ=1)
gender_ = {
    'ชาย': 3,
    'หญิง': 2,
    'อื่นๆ': 1
}
df['gender'] = df['gender'].map(gender_).fillna(3)
print("\n---- สรุปจำนวนเพศ ----")
print(df['gender'].value_counts().sort_index())

# สถานภาพสมรส: แปลงเป็นตัวเลข ordinal
status_ = {
    'โสด ไม่มีแฟน': 4,
    'มีแฟนแต่ยังไม่แต่งงาน': 3,
    'แต่งงานแล้ว': 2,
    'หย่าร้าง/เป็นม่าย': 1
}
df['status'] = df['status'].map(status_).fillna(3)
print("\n---- สรุปจำนวนสถานภาพสมรส----")
print(df['status'].value_counts().sort_index())

# =============================================================================
# SECTION 5: TEXT CLASSIFICATION — ความหมายของแมว
# ใช้ keyword matching แยก free-text เป็น 3 category:
#   family = มองแมวเป็นลูก/ครอบครัว
#   friend = มองแมวเป็นเพื่อน/น้อง
#   pet    = มองแมวเป็นสัตว์เลี้ยงทั่วไป
# =============================================================================

def cat_mean(text):
    text = str(text).lower()
    if any(k in text for k in ['ลูก', 'ครอบครัว', 'สมาชิก', 'family', 'everything', 'ทุกอย่าง', 'คนใน', 'ส่วนนึง', 'ขาดไม่ได้']):
        return 'family'
    elif any(k in text for k in ['เพื่อน', 'เหงา', 'น้อง', 'friend', 'น่ารัก', 'เจ้านาย', 'ฮีล', 'ใจ']):
        return 'friend'
    else:
        return 'pet'

df['mean_2'] = df['cat_mean_2'].apply(cat_mean)
print("\n---- ความหมายน้องแมวของคนเลี้ยง ----")
print(df['mean_2'].value_counts())

# =============================================================================
# SECTION 6: TEXT CLASSIFICATION — พันธุ์แมว
# แบ่งเป็น inter/thai/mixed — คน 1 คนสามารถเลี้ยงหลายพันธุ์ได้
# ใช้ list เก็บ แล้วค่อย explode ทีหลังเพื่อนับ
# =============================================================================

def cat_kind(text):
    text = str(text).lower()
    found = []
    if any(k in text for k in ['เปอร์เซีย', 'persian', 'british', 'บริติช', 'ragdoll', 'scottish',
                                 'scotish', 'สก็อต', 'สก๊อต', 'สกอตต', 'munchkin', 'มัชกิ้น', 'มัชกิน',
                                 'short', 'hair', 'shorthair', 'เมนคูน', 'dsh', 'แรคดอล',
                                 'หิมาลายัน', 'exotic', 'อเมริกัน', 'american', 'ragamuffin']):
        found.append('inter')
    if any(k in text for k in ['ไทย', ' thai', 'วิเชียร', 'ขาวมณี', 'โกนจา', 'ศุภลักษณ์', 'โคราช', 'สีสวาท', 'พื้นเมือง']):
        found.append('thai')
    if any(k in text for k in ['ผสม', 'mix', 'จร', 'บ้าน', 'สลิด', 'ทาง', 'วัด', 'เก็บมาเลี้ยง', 'สามสี', 'ลายปลานิล', 'ธรรมดา']):
        found.append('mixed')
    return found if len(found) > 0 else ['other']

df['kind_list'] = df['cat_kind_3'].apply(cat_kind)
df_explode_3 = df.explode('kind_list')
print("\n---- ประเภทแมวทั้งหมด ----")
print(df_explode_3['kind_list'].value_counts())

# =============================================================================
# SECTION 7: TEXT CLASSIFICATION — แบรนด์อาหารแมว (แบ่งตามช่วงราคา)
#   premium = 250+ บาท/กก
#   medium  = 120-250 บาท/กก
#   low     = ไม่เกิน 120 บาท/กก
# =============================================================================

def brand(text):
    text = str(text).lower()
    found = []
    if any(k in text for k in ['solid gold', 'taste of the wild', 'cheershare', 'เชียร์แชร์',
                                 'bite of wild', 'proplan', 'real power', 'instinct',
                                 'wellness core', 'teste', 'wellnsss']):
        found.append('premium')
    if any(k in text for k in ['king', 'คิง', 'kat', 'wills', 'will', 'purino', 'purina', 'พูริโน',
                                 'ภูริโน', 'projen', 'โปรเจน', 'petheria', 'pateria', 'kativa',
                                 'kaniva', 'แคทิวา', 'แคททิวา', 'แคทิว่า', 'holistic', 'you-o',
                                 'youo', 'ยูโอ', 'buzz', 'perfecta', 'tiffany', 'neez',
                                 'justino', 'mekko', 'smartbrain', 'ostech']):
        found.append('medium')
    if any(k in text for k in ['whiskas', 'wishkas', 'วิสกัส', 'วิสคัส', 'วิสกัต', 'friskies',
                                 'ฟริสกี้', 'โอลิเวอร์', 'ฮีโร่', 'zoi cat', 'betagro',
                                 'เบทราโกร', 'เบทาโกร', 'maxima', 'แม็กซิม่า']):
        found.append('low')
    return found if len(found) > 0 else ['other']

df['brand_list'] = df['brand_4'].apply(brand)
df_exploded_4 = df.explode('brand_list')
print("\n---- การซื้ออาหารแต่ละแบรนด์ในตลาด(แบ่งตามช่วงราคา) ----")
print(df_exploded_4['brand_list'].value_counts())

# =============================================================================
# SECTION 8: LIKERT SCALE — Product Attributes (คำถามที่ 5-9)
# แปลง text → ตัวเลข 1-5
# [หมายเหตุ] fillna(3) = ปานกลาง เพราะ missing น้อย และไม่ต้องการ drop row
# ดู missing rate ใน SECTION 3 สำหรับข้อมูลเพิ่มเติม
# =============================================================================

rating_ = {
    'มากที่สุด': 5,
    'มาก': 4,
    'ปานกลาง': 3,
    'น้อย': 2,
    'น้อยที่สุด': 1
}

df['food_natural_5']   = df['food_natural_5'].map(rating_).fillna(3)
df['food_import_6']    = df['food_import_6'].map(rating_).fillna(3)
df['food_taste_7']     = df['food_taste_7'].map(rating_).fillna(3)
df['food_make_import_8'] = df['food_make_import_8'].map(rating_).fillna(3)
df['brand_popular_9']  = df['brand_popular_9'].map(rating_).fillna(3)

print("\n---- Product Attributes (mean score) ----")
food_cols = ['food_natural_5', 'food_import_6', 'food_taste_7', 'food_make_import_8', 'brand_popular_9']
print(df[food_cols].mean().round(2))

# =============================================================================
# SECTION 9: BINARY — Packaging มีผลต่อการตัดสินใจซื้อ (คำถามที่ 10)
# =============================================================================

rating_1 = {'มีผล': 1, 'ไม่มีผล': 0}
df['packaging_10'] = df['packaging_10'].map(rating_1).fillna(0)
print("\n---- บรรจุภัณฑ์มีผลต่อการตัดสินใจซื้อมั้ย ----")
print(df['packaging_10'].value_counts().sort_index())

# =============================================================================
# SECTION 10: CATEGORICAL — ชอบภาพแบบใดบน packaging (คำถามที่ 11)
# =============================================================================

rating_2 = {
    'ภาพการ์ตูน หรือลายเส้น': 3,
    'ภาพแมวจริง หรือแมวสมจริง (AI)': 2,
    'ได้ทั้งสองแบบ หากถูกใจ': 1
}
df['packaging_11'] = df['packaging_11'].map(rating_2).fillna(1)
print("\n---- บรรจุภัณฑ์(ชอบภาพใด) ----")
print(df['packaging_11'].value_counts().sort_index())

# =============================================================================
# SECTION 11: LIKERT SCALE — Packaging Attributes (คำถามที่ 12-19)
# =============================================================================

pkg_cols = ['packaging_12', 'packaging_13', 'packaging_14', 'packaging_15',
            'packaging_16', 'packaging_17', 'packaging_18', 'packaging_19']
for col in pkg_cols:
    df[col] = df[col].map(rating_).fillna(3)

print("\n---- Packaging Attributes (mean score) ----")
print(df[pkg_cols].mean().round(2))

# =============================================================================
# SECTION 12: TEXT CLASSIFICATION — อยากเพิ่มอะไรใน packaging (คำถามที่ 20)
# แบ่งเป็น: packaging (ซิป/ฝา), quality (สารอาหาร), gift (ของแถม), none, other
# =============================================================================

def add_pkg(text):
    if pd.isnull(text):
        return ['none']
    text = str(text).lower()
    if text.strip() in ['-', 'ไม่มี', 'ไม่รู้', 'ไม่ทราบ', 'ไม่เติม', 'ไม่มีความเห็น', '.', 'none ']:
        return ['none']
    found = []
    if any(k in text for k in ['ซิป', 'zip', 'กล่อง', 'ถุงใส', 'ที่หนีบ', 'ฝา', 'ปิด', 'เปิด', 'เก็บ', 'ถุง']):
        found.append('packaging')
    if any(k in text for k in ['โซเดียม', 'สัญลักษณ์', 'โปรตีน', 'วัตถุดิบ', 'ส่วนผสม', 'สูตร',
                                 'สารอาหาร', 'คุณค่า', 'ไขมัน', 'aafco', 'ปริมาณ', 'คุณสมบัติ',
                                 'วิตามิน', 'สุขภาพ', 'ก้อนขน', 'ไต', 'ตับ', 'ข้อต่อ', 'probiotic',
                                 'ขน', 'โรค', 'ลด']):
        found.append('quality')
    if any(k in text for k in ['ของแถม', 'ของเล่น', 'ลอตเตอรี่', 'ลุ้น', 'สุ่ม', 'topping', 'ก้านขน']):
        found.append('gift')
    return found if len(found) > 0 else ['other']

df['add_pkg_list'] = df['add_pkg_20'].apply(add_pkg)
df_add_pkg = df.explode('add_pkg_list')
print("\n---- สิ่งที่อยากเพิ่มในบรรจุภัณฑ์ ----")
print(df_add_pkg['add_pkg_list'].value_counts())

# =============================================================================
# SECTION 13: LIKERT SCALE — Opinion ต่อแต่ละ Option (1-10) × 5 ด้าน
# รวม 50 columns
# =============================================================================

rating_opinion = {
    'เห็นด้วยที่สุด': 5,
    'เห็นด้วย': 4,
    'เฉยๆ': 3,
    'ไม่เห็นด้วย': 2,
    'ไม่เห็นด้วยเลย': 1
}

option_cols = [
    'opt1_buy', 'opt1_unique', 'opt1_premium', 'opt1_taste', 'opt1_personal',
    'opt2_buy', 'opt2_unique', 'opt2_premium', 'opt2_taste', 'opt2_personal',
    'opt3_buy', 'opt3_unique', 'opt3_premium', 'opt3_taste', 'opt3_personal',
    'opt4_buy', 'opt4_unique', 'opt4_premium', 'opt4_taste', 'opt4_personal',
    'opt5_buy', 'opt5_unique', 'opt5_premium', 'opt5_taste', 'opt5_personal',
    'opt6_buy', 'opt6_unique', 'opt6_premium', 'opt6_taste', 'opt6_personal',
    'opt7_buy', 'opt7_unique', 'opt7_premium', 'opt7_taste', 'opt7_personal',
    'opt8_buy', 'opt8_unique', 'opt8_premium', 'opt8_taste', 'opt8_personal',
    'opt9_buy', 'opt9_unique', 'opt9_premium', 'opt9_taste', 'opt9_personal',
    'opt10_buy', 'opt10_unique', 'opt10_premium', 'opt10_taste', 'opt10_personal'
]

for col in option_cols:
    df[col] = df[col].map(rating_opinion).fillna(3)

# สรุปคะแนนเฉลี่ยแต่ละ Option (ใช้ใน Dashboard)
print("\n---- คะแนนเฉลี่ย opt_buy แต่ละ Option ----")
buy_cols = [f'opt{i}_buy' for i in range(1, 11)]
print(df[buy_cols].mean().round(2).sort_values(ascending=False))

# =============================================================================
# SECTION 14: TOP-3 BINARY FLAGS — Option ไหนถูกเลือกใน Top 3
#
# [แก้ไข BUG] ของเดิมใช้ 'Option 1,|Option 1$' ซึ่งพลาด case ที่ Option 1
# อยู่กลาง string เช่น "Option 3, Option 1, Option 7"
# แก้เป็น regex \bOption 1\b (word boundary) เพื่อให้ match ทุก position
# และป้องกัน Option 1 ชน Option 10 อย่างถูกต้อง
# =============================================================================

for i in range(1, 11):
    df[f'top3_opt{i}'] = df['top_3'].str.contains(
        rf'\bOption {i}\b', regex=True, na=False
    ).astype(int)

print("\n---- จำนวนคนที่เลือกแต่ละ Option ใน Top 3 ----")
top3_cols = [f'top3_opt{i}' for i in range(1, 11)]
print(df[top3_cols].sum().sort_values(ascending=False))

# =============================================================================
# SECTION 15: ONE-HOT ENCODING จาก list columns
# แปลงผลจาก text classification ที่เป็น list → binary flag แต่ละ category
# =============================================================================

# ความหมายของแมว (เข้ารหัส ordinal)
mean_num = {'family': 3, 'friend': 2, 'pet': 1}
df['mean_2'] = df['mean_2'].map(mean_num)

# พันธุ์แมว
df['cat_thai']  = df['kind_list'].apply(lambda x: 1 if 'thai' in str(x) else 0)
df['cat_inter'] = df['kind_list'].apply(lambda x: 1 if 'inter' in str(x) else 0)
df['cat_mixed'] = df['kind_list'].apply(lambda x: 1 if 'mixed' in str(x) else 0)

# ระดับราคาแบรนด์ที่ใช้
df['price_premium'] = df['brand_list'].apply(lambda x: 1 if 'premium' in str(x) else 0)
df['price_medium']  = df['brand_list'].apply(lambda x: 1 if 'medium' in str(x) else 0)
df['price_low']     = df['brand_list'].apply(lambda x: 1 if 'low' in str(x) else 0)

# สิ่งที่อยากเพิ่มใน packaging
df['pkg_none']      = df['add_pkg_list'].apply(lambda x: 1 if 'none' in str(x) else 0)
df['pkg_quality']   = df['add_pkg_list'].apply(lambda x: 1 if 'quality' in str(x) else 0)
df['pkg_packaging'] = df['add_pkg_list'].apply(lambda x: 1 if 'packaging' in str(x) else 0)
df['pkg_gift']      = df['add_pkg_list'].apply(lambda x: 1 if 'gift' in str(x) else 0)
df['pkg_other']     = df['add_pkg_list'].apply(lambda x: 1 if 'other' in str(x) else 0)

# =============================================================================
# SECTION 16: DROP COLUMNS ที่ใช้งานเสร็จแล้ว / ไม่ใช้ใน ML
# =============================================================================

drop_cols = [
    'had_none_1', 'cat_mean_2', 'cat_kind_3', 'brand_4', 'add_pkg_20',
    'num_1', 'top_3',
    'เมื่อนึกถึงบรรจุภัณฑ์อาหารแมวที่โดดเด่นและสะดุดตา คุณคิดถึงแบรนด์ใดเป็นอันดับแรก เพราะอะไร [โปรดพิมพ์ชื่อแบรนด์และเหตุผล]',
    'kind_list', 'brand_list', 'add_pkg_list'
]
df = df.drop(columns=drop_cols)

# =============================================================================
# SECTION 17: FINAL SUMMARY
# =============================================================================

print("\n---- Dataset Summary ----")
print(f"จำนวน rows (คนที่เคยซื้ออาหารแมว): {len(df)}")
print(f"จำนวน columns: {len(df.columns)}")
print(f"Missing values คงเหลือ: {df.isnull().sum().sum()}")
print("\nColumn ทั้งหมด:")
print(list(df.columns))

# =============================================================================
# SECTION 18: FEATURE SCALING DOCUMENTATION
#
# [หมายเหตุสำคัญสำหรับรายงาน Data Preprocessing]
#
# ในขั้นตอนนี้ยังไม่ได้ทำ Feature Scaling กับ clean_cat.csv
# เพราะการ Scale ใน clean dataset โดยตรงมีความเสี่ยง Data Leakage:
#   - ถ้า StandardScaler fit บน data ทั้งหมด (รวม test set)
#     → โมเดลจะ "รู้" ข้อมูลของ test set ล่วงหน้า
#     → ผลประเมินโมเดลจะดูดีเกินจริง (Optimistic Bias)
#
# วิธีที่ถูกต้อง (ใช้ใน 4_supervised.py):
#   Pipeline([('scaler', StandardScaler()), ('model', ...)])
#   → Scaler จะ fit เฉพาะ X_train แล้วค่อย transform X_test
#   → ป้องกัน Data Leakage ได้อย่างสมบูรณ์
#
# Feature ที่ต้องการ Scaling (Numeric, range ต่างกัน):
#   - food_natural_5, food_import_6, food_taste_7, food_make_import_8, brand_popular_9
#     → Likert 1-5 (ต้องการ scale)
#   - packaging_12 ถึง packaging_19
#     → Likert 1-5 (ต้องการ scale)
#   - age_, gender, status, mean_2
#     → Ordinal 1-4 (ต้องการ scale)
#
# Feature ที่ไม่ต้องการ Scaling (Binary 0/1):
#   - cat_thai, cat_inter, cat_mixed
#   - price_premium, price_medium, price_low
#   - pkg_none, pkg_quality, pkg_packaging, pkg_gift, pkg_other
#   - top3_opt1 ถึง top3_opt10
# =============================================================================

from sklearn.preprocessing import StandardScaler

# แสดง range ของ numeric features ก่อน scale (สำหรับรายงาน)
numeric_features = [
    'food_natural_5', 'food_import_6', 'food_taste_7', 'food_make_import_8', 'brand_popular_9',
    'packaging_12', 'packaging_13', 'packaging_14', 'packaging_15',
    'packaging_16', 'packaging_17', 'packaging_18', 'packaging_19',
    'age_', 'gender', 'status', 'mean_2'
]
numeric_features = [c for c in numeric_features if c in df.columns]

print("\n---- Feature Scaling Report (Before Scale) ----")
print(f"{'Feature':<25} {'Min':>6} {'Max':>6} {'Mean':>8} {'Std':>8}")
print("-" * 58)
for col in numeric_features:
    print(f"{col:<25} {df[col].min():>6.2f} {df[col].max():>6.2f} {df[col].mean():>8.2f} {df[col].std():>8.2f}")

print("\n[หมายเหตุ] StandardScaler จะถูกใช้ใน ML Pipeline (4_supervised.py, 3_eda_unsupervised.py)")
print("          ไม่ scale ใน clean_cat.csv เพื่อป้องกัน Data Leakage")

# =============================================================================
# SECTION 19: EXPORT
# clean_cat.csv = dataset สำหรับทีม ML (Unsupervised + Supervised)
# StandardScaler ทำใน Pipeline ของแต่ละ model script — ไม่ทำที่นี่
# =============================================================================

df.to_csv('data/processed/clean_cat.csv', index=False, encoding='utf-8-sig')
print("\n✅ Export data/processed/clean_cat.csv เรียบร้อย")