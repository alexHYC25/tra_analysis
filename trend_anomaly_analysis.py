import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 0. 設定 Matplotlib 支援顯示繁體中文 (避免圖表出現方塊)
# ==========================================
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] # Windows 使用微軟正黑體
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 從 MySQL 撈取特定車站的時間序列資料
# ==========================================
print("📥 正在從資料庫撈取時間序列資料...")
db_url = 'mysql+pymysql://root:@localhost:3306/TRA_DataMining'
engine = create_engine(db_url)

# 挑選對比強烈的車站：通勤大站(桃園) vs 觀光大站(瑞芳)
target_stations = "('桃園', '瑞芳')"

query = f"""
    SELECT 
        d.trnOpDate AS date, 
        s.staName AS station, 
        (d.gateInComingCnt + d.gateOutGoingCnt) AS total_passengers
    FROM 
        DailyPassenger d
    JOIN 
        StationInfo s ON d.staCode = s.staCode
    WHERE 
        s.staName IN {target_stations}
    ORDER BY 
        d.trnOpDate ASC;
"""

df = pd.read_sql(query, con=engine)
df['date'] = pd.to_datetime(df['date'])

# ==========================================
# 2. 資料處理與異常偵測 (Z-Score 演算法)
# ==========================================
print("🤖 正在計算移動平均趨勢與偵測異常值...")

# 設定畫布大小，準備畫兩張子圖 (一個車站一張)
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 10), sharex=True)
stations = df['station'].unique()

for i, station in enumerate(stations):
    # 取出單一車站資料
    st_df = df[df['station'] == station].copy()
    st_df.set_index('date', inplace=True)
    
    # 【分析一：趨勢平滑】計算 30 天移動平均線 (消弭短期波動，看長期趨勢)
    st_df['30D_MA'] = st_df['total_passengers'].rolling(window=30).mean()
    
    # 【分析二：異常偵測】使用 Z-Score 找出極端值
    # Z-Score = (當日運量 - 平均運量) / 標準差。絕對值大於 3 通常視為極端異常
    mean_pax = st_df['total_passengers'].mean()
    std_pax = st_df['total_passengers'].std()
    st_df['z_score'] = (st_df['total_passengers'] - mean_pax) / std_pax
    
    # 標記異常突波 (Z-score > 3 代表超乎尋常的爆滿)
    anomalies = st_df[st_df['z_score'] > 3]
    
    # ==========================================
    # 3. 繪製精美圖表
    # ==========================================
    ax = axes[i]
    
    # 畫出原始每日運量 (用較淡的顏色作為背景)
    ax.plot(st_df.index, st_df['total_passengers'], color='gray', alpha=0.3, label='每日實際運量')
    
    # 畫出 30 天移動平均線 (明顯的趨勢線)
    ax.plot(st_df.index, st_df['30D_MA'], color='blue' if i==0 else 'green', linewidth=2, label='30天移動平均趨勢')
    
    # 標記異常爆滿的日子 (用紅色散佈點突顯)
    ax.scatter(anomalies.index, anomalies['total_passengers'], color='red', s=50, zorder=5, label='異常突波 (Z-Score > 3)')
    
    ax.set_title(f'{station}車站：2019-2025 運量趨勢與異常偵測', fontsize=16, fontweight='bold')
    ax.set_ylabel('進出站總人數', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right')

plt.tight_layout()
plt.show()
print("🎉 圖表繪製完成！")