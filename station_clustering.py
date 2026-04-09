import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import sys
import warnings

# Windows 終端機相容 UTF-8 輸出（避免 emoji 錯誤）
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

warnings.filterwarnings('ignore')

# ==========================================
# 1. 從 MySQL 讀取特徵視圖
# ==========================================
print("📥 正在從資料庫讀取車站特徵...")
db_url = 'mysql+pymysql://root:@localhost:3306/TRA_DataMining'
engine = create_engine(db_url)

df = pd.read_sql("SELECT * FROM v_StationFeatures WHERE avg_daily_total > 0;", con=engine)
df = df.dropna()

# 統一欄位命名（相容中英文欄位名稱的 View）
df.columns = ['staCode', 'staName', 'avg_daily_total', 'avg_weekend_total', 'avg_weekday_total', 'weekend_weekday_ratio']
df = df[df['staName'] != '枋野'].reset_index(drop=True)
df['avg_daily_total'] = df['avg_daily_total'].astype(float)
df['weekend_weekday_ratio'] = df['weekend_weekday_ratio'].astype(float)

print(f"  ✅ 讀取完成：共 {len(df)} 個有效車站\n")

# ==========================================
# 2. 特徵工程：標準化 (Standardization)
# ==========================================
features = ['avg_daily_total', 'weekend_weekday_ratio']
X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================================
# 3. K-Means 分群模型訓練
# ==========================================
print("🤖 正在訓練 K-Means 分群模型...\n")
K = 4
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
df['cluster_id'] = kmeans.fit_predict(X_scaled)

# ==========================================
# 4. 自動貼上語意標籤
#    根據分群中心點的「日均運量」與「平假日比例」
#    分成四個象限，自動命名
# ==========================================
centers_orig = scaler.inverse_transform(kmeans.cluster_centers_)
center_df = pd.DataFrame(centers_orig, columns=features)
center_df['cluster_id'] = range(K)

# 台鐵資料中幾乎所有車站假日比均 > 1，因此使用「群組間相對排名」
# 而非絕對門檻來決定標籤，避免全部被貼成同一類。
#
# 策略：
#   按 avg_daily_total 排名 → 前 2 名為「大站」，後 2 名為「小站」
#   在大站組 / 小站組內，再按 weekend_weekday_ratio 排名
#   大站 + 較高比值 → 都會觀光熱站；大站 + 較低比值 → 都會通勤大站
#   小站 + 較高比值 → 週末觀光熱點；小站 + 較低比值 → 在地一般站

center_sorted = center_df.sort_values('avg_daily_total', ascending=False).reset_index(drop=True)
# volume_rank: 0 = 最大, 1 = 次大, …
center_sorted['volume_rank'] = center_sorted.index

cluster_name_map = {}
for _, row in center_sorted.iterrows():
    cid   = int(row['cluster_id'])
    vrank = int(row['volume_rank'])

    if vrank == 0:
        cluster_name_map[cid] = '都會核心大站'
    elif vrank == 1:
        cluster_name_map[cid] = '都會通勤大站'
    else:
        # 在剩餘 2 個小站群中，ratio 最高的視為觀光屬性
        low_two = center_sorted[center_sorted['volume_rank'] >= 2].sort_values(
            'weekend_weekday_ratio', ascending=False
        )
        if cid == int(low_two.iloc[0]['cluster_id']):
            cluster_name_map[cid] = '週末觀光熱點'
        else:
            cluster_name_map[cid] = '在地一般站'

df['cluster_name'] = df['cluster_id'].map(cluster_name_map)

# ==========================================
# 5. 將分群結果儲存回資料庫
# ==========================================
print("💾 正在將分群結果儲存回資料庫...")
with engine.begin() as conn:
    # 新增欄位（已存在則略過）
    for col_def in [
        "ALTER TABLE StationInfo ADD COLUMN cluster_id INT DEFAULT NULL",
        "ALTER TABLE StationInfo ADD COLUMN cluster_name VARCHAR(20) DEFAULT NULL",
    ]:
        try:
            conn.execute(text(col_def))
        except Exception:
            pass  # 欄位已存在

    # 批次更新每個車站的群組（一次送出所有資料）
    batch = [
        {'cid': int(row['cluster_id']), 'cname': row['cluster_name'], 'code': row['staCode']}
        for _, row in df.iterrows()
    ]
    conn.execute(
        text("UPDATE StationInfo SET cluster_id = :cid, cluster_name = :cname WHERE staCode = :code"),
        batch
    )

print("✅ 分群結果已儲存至資料庫！\n")

# ==========================================
# 6. 印出分群剖析報告
# ==========================================
print("🎉 分群完成！以下是全台車站四大聚落 DNA 分析：\n")
for cid in sorted(cluster_name_map):
    cname = cluster_name_map[cid]
    cluster_data = df[df['cluster_id'] == cid]
    avg_people = int(cluster_data['avg_daily_total'].mean())
    avg_ratio  = round(cluster_data['weekend_weekday_ratio'].mean(), 2)
    count      = len(cluster_data)
    top5 = cluster_data.sort_values('avg_daily_total', ascending=False).head(5)['staName'].tolist()

    print(f"🔹 【{cname}】共 {count} 個車站")
    print(f"   ▶ 日均運量：約 {avg_people:,} 人")
    print(f"   ▶ 平假日比：{avg_ratio}  (>1 = 假日較多)")
    print(f"   ▶ 代表車站：{', '.join(top5)}")
    print("-" * 50)
