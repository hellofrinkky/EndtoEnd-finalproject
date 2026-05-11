import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # ไม่ต้องการ display window — save เป็นไฟล์แทน
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 3_eda_unsupervised.py — EDA + Clustering + PCA
#
# Input : clean_cat.csv (จาก 1_clean.py + 2_target.py)
# Output:
#   - clean_cat.csv (เพิ่ม column 'cluster')
#   - รูป EDA/Clustering ใน folder output_plots/
#
# Pipeline:
#   1. Descriptive Statistics
#   2. Correlation Analysis
#   3. Elbow Method → หา optimal K
#   4. K-Means Clustering
#   5. PCA (2D visualization)
#   6. Customer Persona ต่อ Cluster
# =============================================================================

# ------------------------------------------------------------------
# Setup font สำหรับภาษาไทยใน matplotlib
# ถ้าไม่มี font ภาษาไทย label จะแสดงเป็น □ — ไม่กระทบ logic
# ------------------------------------------------------------------
try:
    thai_fonts = [f for f in fm.findSystemFonts() if any(n in f.lower() for n in ['thsarabun', 'noto', 'tahoma'])]
    if thai_fonts:
        plt.rcParams['font.family'] = fm.FontProperties(fname=thai_fonts[0]).get_name()
except:
    pass

import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.makedirs('output_plots', exist_ok=True)

# =============================================================================
# LOAD DATA
# =============================================================================

df = pd.read_csv('data/processed/clean_cat.csv')
print(f"✅ โหลดข้อมูล: {df.shape[0]} rows, {df.shape[1]} columns")

# =============================================================================
# SECTION 1: DESCRIPTIVE STATISTICS
# แสดง mean, std, min, max ของทุก numeric column
# =============================================================================

print("\n" + "="*60)
print("SECTION 1: DESCRIPTIVE STATISTICS")
print("="*60)

desc = df.describe().T[['mean', 'std', 'min', '50%', 'max']]
desc.columns = ['mean', 'std', 'min', 'median', 'max']
print(desc.round(2).to_string())
desc.round(2).to_csv('output_plots/descriptive_stats.csv')
print("\n✅ บันทึก descriptive_stats.csv")

# =============================================================================
# SECTION 2: CORRELATION ANALYSIS
# ดู correlation ระหว่าง Product/Packaging attributes กับ opt3_buy (winner)
# =============================================================================

print("\n" + "="*60)
print("SECTION 2: CORRELATION ANALYSIS")
print("="*60)

# Feature groups
food_cols    = ['food_natural_5', 'food_import_6', 'food_taste_7', 'food_make_import_8', 'brand_popular_9']
pkg_cols     = ['packaging_12', 'packaging_13', 'packaging_14', 'packaging_15',
                'packaging_16', 'packaging_17', 'packaging_18', 'packaging_19']
opt_buy_cols = [f'opt{i}_buy' for i in range(1, 11)]

# Correlation ของ features กับ opt3_buy
corr_target = df[food_cols + pkg_cols].corrwith(df['opt3_buy']).sort_values(ascending=False)
print("\nCorrelation กับ opt3_buy (คะแนน อยากซื้อ Option 3):")
print(corr_target.round(3).to_string())

# Plot heatmap ของ packaging attributes
fig, ax = plt.subplots(figsize=(10, 8))
corr_matrix = df[pkg_cols + food_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, ax=ax, linewidths=0.5)
ax.set_title('Correlation Heatmap: Product & Packaging Attributes', fontsize=14, pad=15)
plt.tight_layout()
plt.savefig('output_plots/correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ บันทึก correlation_heatmap.png")

# Plot opt_buy คะแนนเฉลี่ยแต่ละ Option (Bar chart)
opt_means = df[opt_buy_cols].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10, 5))
colors = ['#e74c3c' if 'opt3' in c else '#3498db' for c in opt_means.index]
bars = ax.bar(range(1, 11), opt_means.values, color=colors)
ax.set_xticks(range(1, 11))
ax.set_xticklabels([f'Opt {i}' for i in range(1, 11)])
ax.set_ylabel('Mean Buy Score (1-5)')
ax.set_title('Average "Buy Intent" Score per Packaging Option\n(Red = Recommended Option 3)', fontsize=13)
ax.set_ylim(0, 5)
for bar, val in zip(bars, opt_means.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f'{val:.2f}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig('output_plots/option_buy_scores.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ บันทึก option_buy_scores.png")

# =============================================================================
# SECTION 3: FEATURE SELECTION สำหรับ Clustering
# ใช้ food + packaging attributes เป็น features
# เหตุผล: สะท้อน preference ของลูกค้าได้ตรงที่สุด
# ไม่ใช้ option scores เพราะจะทำให้ cluster bias ไปที่ option เดียว
# =============================================================================

cluster_features = food_cols + pkg_cols
X = df[cluster_features].copy()

print(f"\nFeatures สำหรับ Clustering: {len(cluster_features)} columns")
print(cluster_features)

# StandardScaler — จำเป็นสำหรับ K-Means เพราะ scale ต่างกัน
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =============================================================================
# SECTION 4: ELBOW METHOD — หา optimal K
# ดู inertia (WCSS) และ silhouette score ที่ K=2 ถึง 8
# =============================================================================

print("\n" + "="*60)
print("SECTION 4: ELBOW METHOD")
print("="*60)

inertias   = []
sil_scores = []
K_range    = range(2, 9)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, km.labels_))
    print(f"  K={k}: Inertia={km.inertia_:.1f}, Silhouette={sil_scores[-1]:.3f}")

# Plot Elbow + Silhouette แบบ side-by-side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(list(K_range), inertias, 'bo-', linewidth=2, markersize=8)
ax1.set_xlabel('Number of Clusters (K)')
ax1.set_ylabel('Inertia (WCSS)')
ax1.set_title('Elbow Method')
ax1.grid(True, alpha=0.3)

ax2.plot(list(K_range), sil_scores, 'rs-', linewidth=2, markersize=8)
ax2.set_xlabel('Number of Clusters (K)')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette Score')
ax2.grid(True, alpha=0.3)

plt.suptitle('Optimal K Selection', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('output_plots/elbow_silhouette.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ บันทึก elbow_silhouette.png")

# เลือก K ที่ให้ Silhouette สูงสุด
best_k = list(K_range)[sil_scores.index(max(sil_scores))]
print(f"\n✅ Optimal K = {best_k} (Silhouette = {max(sil_scores):.3f})")

# =============================================================================
# SECTION 5: K-MEANS CLUSTERING
# Train ด้วย optimal K และเพิ่ม cluster label กลับ DataFrame
# =============================================================================

print("\n" + "="*60)
print(f"SECTION 5: K-MEANS (K={best_k})")
print("="*60)

km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df['cluster'] = km_final.fit_predict(X_scaled)

print("\nจำนวนสมาชิกแต่ละ Cluster:")
print(df['cluster'].value_counts().sort_index())

# Cluster Profiling — mean ของ features แต่ละ cluster
cluster_profile = df.groupby('cluster')[cluster_features].mean().round(2)
print("\nCluster Profile (mean score):")
print(cluster_profile.to_string())
cluster_profile.to_csv('output_plots/cluster_profile.csv')
print("✅ บันทึก cluster_profile.csv")

# Radar chart — cluster profile
def radar_chart(data, title, filename):
    categories = list(data.columns)
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

    for i, (idx, row) in enumerate(data.iterrows()):
        values = row.tolist() + [row.tolist()[0]]
        ax.plot(angles, values, 'o-', linewidth=2, color=colors[i % len(colors)],
                label=f'Cluster {idx}')
        ax.fill(angles, values, alpha=0.1, color=colors[i % len(colors)])

    ax.set_xticks(angles[:-1])
    short_labels = ['Natural', 'Import', 'Taste', 'ForeignMade', 'Brand',
                    'Premium', 'CatPic', 'FoodPic', 'Ingredient', 'Eco',
                    'Origin', 'Function', 'Guarantee'][:N]
    ax.set_xticklabels(short_labels, size=9)
    ax.set_ylim(1, 5)
    ax.set_title(title, size=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(f'output_plots/{filename}', dpi=150, bbox_inches='tight')
    plt.close()

radar_chart(cluster_profile, f'Cluster Profile Radar (K={best_k})', 'cluster_radar.png')
print("✅ บันทึก cluster_radar.png")

# =============================================================================
# SECTION 6: PCA — 2D VISUALIZATION
# ลด dimension เพื่อ plot cluster บน 2D
# =============================================================================

print("\n" + "="*60)
print("SECTION 6: PCA (2D)")
print("="*60)

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

var_explained = pca.explained_variance_ratio_
print(f"PC1 explains: {var_explained[0]:.1%}")
print(f"PC2 explains: {var_explained[1]:.1%}")
print(f"Total variance explained: {sum(var_explained):.1%}")

# Plot PCA scatter
fig, ax = plt.subplots(figsize=(9, 7))
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
for cluster_id in sorted(df['cluster'].unique()):
    mask = df['cluster'] == cluster_id
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c=colors[cluster_id % len(colors)],
               label=f'Cluster {cluster_id}',
               alpha=0.7, s=80, edgecolors='white', linewidth=0.5)

# Plot centroids
centroids_pca = pca.transform(km_final.cluster_centers_)
ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
           c='black', marker='X', s=200, zorder=5, label='Centroids')

ax.set_xlabel(f'PC1 ({var_explained[0]:.1%} variance)', fontsize=11)
ax.set_ylabel(f'PC2 ({var_explained[1]:.1%} variance)', fontsize=11)
ax.set_title(f'K-Means Clustering Visualization (PCA 2D)\nK={best_k}', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('output_plots/pca_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ บันทึก pca_scatter.png")

# PCA feature loadings — ดูว่า feature ไหนส่งผลต่อ PC1 มากที่สุด
loadings = pd.DataFrame(
    pca.components_.T,
    index=cluster_features,
    columns=['PC1', 'PC2']
).round(3)
print("\nPCA Loadings (top features):")
print(loadings.sort_values('PC1', ascending=False).head(5).to_string())

# =============================================================================
# SECTION 7: CUSTOMER PERSONA
# ตีความแต่ละ Cluster จาก profile และ demographics
# =============================================================================

print("\n" + "="*60)
print("SECTION 7: CUSTOMER PERSONA")
print("="*60)

persona_demo = df.groupby('cluster').agg(
    count=('gender', 'count'),
    gender_mode=('gender', lambda x: x.mode()[0]),
    age_mean=('age_', 'mean'),
    status_mode=('status', lambda x: x.mode()[0]),
    cat_meaning=('mean_2', 'mean'),
    packaging_matters=('packaging_10', 'mean'),
    opt3_buy_mean=('opt3_buy', 'mean'),
).round(2)

print(persona_demo.to_string())
persona_demo.to_csv('output_plots/persona_demo.csv')

# สรุป persona อัตโนมัติตาม score สูงสุด
print("\n---- สรุป Persona ต่อ Cluster ----")
for cid in sorted(df['cluster'].unique()):
    row = cluster_profile.loc[cid]
    top_food = row[food_cols].idxmax()
    top_pkg  = row[pkg_cols].idxmax()
    opt3_mean = persona_demo.loc[cid, 'opt3_buy_mean']
    count    = persona_demo.loc[cid, 'count']

    food_label = {
        'food_natural_5': 'วัตถุดิบธรรมชาติ',
        'food_import_6': 'วัตถุดิบนำเข้า',
        'food_taste_7': 'รสชาติ',
        'food_make_import_8': 'สินค้าต่างประเทศ',
        'brand_popular_9': 'แบรนด์ดัง'
    }
    pkg_label = {
        'packaging_12': 'ดูพรีเมียม',
        'packaging_13': 'มีภาพแมว',
        'packaging_14': 'มีภาพอาหารเม็ด',
        'packaging_15': 'มีภาพวัตถุดิบ',
        'packaging_16': 'Eco-friendly',
        'packaging_17': 'บอกแหล่งผลิต',
        'packaging_18': 'บอก Functional Benefit',
        'packaging_19': 'มีรางวัล/การันตี'
    }

    print(f"\nCluster {cid} ({count} คน | opt3_buy avg={opt3_mean:.2f})")
    print(f"  ให้ความสำคัญสูงสุด: {food_label.get(top_food, top_food)}")
    print(f"  ชอบ packaging ที่: {pkg_label.get(top_pkg, top_pkg)}")

# =============================================================================
# SECTION 8: ANOMALY DETECTION — Isolation Forest
#
# เหตุผลที่ทำ Anomaly Detection:
#   - ตรวจหาผู้ตอบแบบสอบถามที่ตอบผิดปกติ เช่น กด 1 ทุกข้อ หรือ 5 ทุกข้อ
#   - ข้อมูล anomalous อาจทำให้ Clustering และ Model เบี่ยงเบนได้
#   - Isolation Forest เหมาะกับ survey data เพราะไม่ต้องการ label
#
# ผลที่ได้:
#   - anomaly = -1 → ผู้ตอบที่ผิดปกติ (outlier)
#   - anomaly =  1 → ผู้ตอบปกติ
# =============================================================================

from sklearn.ensemble import IsolationForest

print("\n" + "="*60)
print("SECTION 8: ANOMALY DETECTION (Isolation Forest)")
print("="*60)

# ใช้ features เดิมที่ใช้ Clustering
iso = IsolationForest(
    contamination=0.05,  # สมมติว่า ~5% ของ data อาจเป็น outlier
    random_state=42,
    n_estimators=100
)
anomaly_labels = iso.fit_predict(X_scaled)  # -1 = anomaly, 1 = normal
anomaly_scores = iso.decision_function(X_scaled)  # ยิ่งต่ำยิ่ง anomalous

df['anomaly']       = anomaly_labels
df['anomaly_score'] = anomaly_scores.round(4)

n_anomaly = (anomaly_labels == -1).sum()
print(f"\nจำนวน Anomaly ที่ตรวจพบ: {n_anomaly} คน ({n_anomaly/len(df)*100:.1f}%)")
print(f"จำนวน Normal: {(anomaly_labels == 1).sum()} คน")

# แสดง anomaly rows
anomaly_rows = df[df['anomaly'] == -1][cluster_features + ['anomaly_score', 'cluster']]
print("\nตัวอย่าง Anomaly rows (score ต่ำสุด = ผิดปกติที่สุด):")
print(anomaly_rows.sort_values('anomaly_score').head(5).to_string())

# Plot anomaly บน PCA scatter
fig, ax = plt.subplots(figsize=(9, 7))
normal_mask  = anomaly_labels == 1
anomaly_mask = anomaly_labels == -1

ax.scatter(X_pca[normal_mask, 0], X_pca[normal_mask, 1],
           c='#3498db', alpha=0.6, s=70, label='Normal', edgecolors='white', linewidth=0.5)
ax.scatter(X_pca[anomaly_mask, 0], X_pca[anomaly_mask, 1],
           c='#e74c3c', alpha=0.9, s=120, marker='X', label='Anomaly', zorder=5)

ax.set_xlabel(f'PC1 ({var_explained[0]:.1%} variance)', fontsize=11)
ax.set_ylabel(f'PC2 ({var_explained[1]:.1%} variance)', fontsize=11)
ax.set_title('Anomaly Detection — Isolation Forest\n(Red X = Anomaly, Blue = Normal)', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('output_plots/anomaly_detection.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ บันทึก anomaly_detection.png")

# Export anomaly summary
anomaly_summary = pd.DataFrame({
    'total_respondents': [len(df)],
    'normal_count':  [(anomaly_labels == 1).sum()],
    'anomaly_count': [n_anomaly],
    'anomaly_pct':   [round(n_anomaly/len(df)*100, 1)],
    'contamination_setting': [0.05]
})
anomaly_summary.to_csv('output_plots/anomaly_summary.csv', index=False)
print("✅ บันทึก anomaly_summary.csv")

print("\n[หมายเหตุ] Anomaly rows ยังคงอยู่ใน dataset")
print("  → ไม่ drop ออกเพราะ sample มีเพียง 148 rows")
print("  → ใช้ column 'anomaly' เป็น flag สำหรับ sensitivity analysis")

# =============================================================================
# SECTION 9: EXPORT — เพิ่ม cluster + anomaly label กลับ clean_cat.csv
# ทีม Supervised จะใช้ column 'cluster' เป็น feature เพิ่มเติม
# =============================================================================

df.to_csv('data/processed/clean_cat.csv', index=False, encoding='utf-8-sig')
print("\n✅ Export data/processed/clean_cat.csv พร้อม column 'cluster' เรียบร้อย")
print("   → ส่งต่อให้ ML Engineer Supervised (คนที่ 4) ได้เลย")

print("\n" + "="*60)
print("Output files ใน output_plots/:")
for f in sorted(os.listdir('output_plots')):
    print(f"  - {f}")
print("="*60)