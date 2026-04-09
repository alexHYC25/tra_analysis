"""
台鐵營運智慧分析系統 — 戰情儀表板 v2
執行方式：python -m streamlit run dashboard.py
"""
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine, text

# ============================================================
# 頁面基礎設定
# ============================================================
st.set_page_config(
    page_title="台鐵營運智慧分析系統",
    page_icon="🚂",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 雲端部署時從環境變數或 st.secrets 讀取，本機開發則使用 localhost
import os
DB_URL = (
    os.environ.get("DB_URL")
    or st.secrets.get("DB_URL", None)
    or 'mysql+pymysql://root:@localhost:3306/TRA_DataMining'
)

CLUSTER_COLORS = {
    '都會核心大站': '#d62728',
    '都會通勤大站': '#1f77b4',
    '週末觀光熱點': '#ff7f0e',
    '在地一般站':   '#2ca02c',
}
CLUSTER_ICONS = {
    '都會核心大站': '🏙️',
    '都會通勤大站': '🚉',
    '週末觀光熱點': '🌊',
    '在地一般站':   '🌾',
}

# ============================================================
# 共用工具
# ============================================================
def fmt_number(n):
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(int(n))

def anomaly_detect(series, window=30, threshold=3.0):
    rm = series.rolling(window, min_periods=window//2).mean()
    rs = series.rolling(window, min_periods=window//2).std()
    z  = (series - rm) / rs.replace(0, np.nan)
    return z.abs() > threshold, z

@st.cache_data(ttl=600)
def get_taiwan_holidays():
    from workalendar.asia import Taiwan
    cal = Taiwan()
    rows = []
    for year in range(2019, 2027):
        for date, name in cal.holidays(year):
            rows.append({'ds': pd.Timestamp(date), 'holiday': name,
                         'holiday_name': name})
    return pd.DataFrame(rows)

# ============================================================
# 車種別資料解析
# ============================================================
@st.cache_data
def load_train_type_data():
    import os
    csv_path = os.path.join(os.path.dirname(__file__), '2019-2025車種別客運量.csv')
    raw = pd.read_csv(csv_path, header=None, encoding='utf-8-sig')

    month_map = {'Jan.':1,'Feb.':2,'Mar.':3,'Apr.':4,'May':5,'June':6,
                 'July':7,'Aug.':8,'Sep.':9,'Oct.':10,'Nov.':11,'Dec.':12}

    def to_num(v):
        if pd.isna(v) or str(v).strip() in ('', '-'):
            return 0
        return int(str(v).replace(',', '').strip())

    records = []
    current_year = None
    for _, row in raw.iterrows():
        val1 = str(row[1]).strip() if not pd.isna(row[1]) else ''
        # 年份列
        if val1.isdigit() and len(val1) == 4:
            current_year = int(val1)
            continue
        # 月份列
        if val1 in month_map and current_year:
            records.append({
                'year': current_year,
                'month': month_map[val1],
                'ym': pd.Timestamp(f'{current_year}-{month_map[val1]:02d}-01'),
                'pax_total':    to_num(row[2]),
                'pax_tzechiang': to_num(row[3]),
                'pax_chukuang': to_num(row[4]),
                'pax_local':    to_num(row[5]),
                'pax_ordinary': to_num(row[6]),
                'km_total':     to_num(row[7]),
                'km_tzechiang': to_num(row[8]),
                'km_chukuang':  to_num(row[9]),
                'km_local':     to_num(row[10]),
                'km_ordinary':  to_num(row[11]),
            })
    df = pd.DataFrame(records)
    df = df[df['pax_total'] > 0].reset_index(drop=True)
    return df

# ============================================================
# 資料讀取函式
# ============================================================
@st.cache_data(ttl=300)
def load_cluster_data():
    engine = create_engine(DB_URL)
    sta = pd.read_sql(
        "SELECT staCode, staName, cluster_id, cluster_name, lat, lon, city "
        "FROM stationinfo WHERE cluster_id IS NOT NULL AND staName != '枋野'",
        con=engine
    )
    feat_raw = pd.read_sql(
        "SELECT * FROM v_StationFeatures WHERE avg_daily_total > 0", con=engine
    )
    cols = list(feat_raw.columns); cols[0]='staCode'; cols[1]='_n'
    feat_raw.columns = cols
    feat = feat_raw[['staCode','avg_daily_total','avg_weekend_total',
                     'avg_weekday_total','weekend_weekday_ratio']]
    df = sta.merge(feat, on='staCode', how='inner')
    for c in ['avg_daily_total','avg_weekend_total','avg_weekday_total','weekend_weekday_ratio']:
        df[c] = df[c].astype(float)
    return df.sort_values('avg_daily_total', ascending=False).reset_index(drop=True)

@st.cache_data(ttl=300)
def load_station_list():
    engine = create_engine(DB_URL)
    return pd.read_sql(
        "SELECT staCode, staName FROM stationinfo WHERE staName IS NOT NULL ORDER BY staName",
        con=engine
    )

@st.cache_data(ttl=300)
def load_station_timeseries(sta_code):
    engine = create_engine(DB_URL)
    q = text("SELECT trnOpDate AS date, "
             "(gateInComingCnt+gateOutGoingCnt) AS total_passengers "
             "FROM dailypassenger WHERE staCode=:code ORDER BY trnOpDate")
    df = pd.read_sql(q, con=engine, params={'code': sta_code})
    df['date'] = pd.to_datetime(df['date'])
    return df

@st.cache_data(ttl=300)
def load_multi_station_timeseries(codes_tuple):
    engine = create_engine(DB_URL)
    placeholders = ','.join([f"'{c}'" for c in codes_tuple])
    q = f"""SELECT d.trnOpDate AS date, s.staName AS station,
                   (d.gateInComingCnt+d.gateOutGoingCnt) AS total_passengers
            FROM dailypassenger d JOIN stationinfo s ON d.staCode=s.staCode
            WHERE d.staCode IN ({placeholders}) ORDER BY date"""
    df = pd.read_sql(q, con=engine)
    df['date'] = pd.to_datetime(df['date'])
    return df

@st.cache_data(ttl=300)
def load_monthly_total():
    engine = create_engine(DB_URL)
    df = pd.read_sql(
        "SELECT DATE_FORMAT(trnOpDate,'%%Y-%%m') AS ym, "
        "SUM(gateInComingCnt+gateOutGoingCnt) AS total "
        "FROM dailypassenger GROUP BY ym ORDER BY ym",
        con=engine
    )
    df['ym'] = pd.to_datetime(df['ym'])
    return df

@st.cache_data(ttl=300)
def load_recent_anomalies(days=90):
    engine = create_engine(DB_URL)
    q = text("""SELECT d.trnOpDate AS date, s.staName AS station,
                       s.cluster_name,
                       (d.gateInComingCnt+d.gateOutGoingCnt) AS total
               FROM dailypassenger d JOIN stationinfo s ON d.staCode=s.staCode
               WHERE d.trnOpDate >= DATE_SUB(CURDATE(), INTERVAL :days DAY)""")
    df = pd.read_sql(q, con=engine, params={'days': days})
    df['date'] = pd.to_datetime(df['date'])
    df['z'] = df.groupby('station')['total'].transform(
        lambda x: (x - x.mean()) / x.std()
    )
    return df[df['z'] > 3].sort_values('z', ascending=False)

@st.cache_data(ttl=300)
def load_yearly_totals():
    engine = create_engine(DB_URL)
    q = """SELECT s.staCode, s.staName, s.cluster_name,
                  YEAR(d.trnOpDate) AS yr,
                  SUM(d.gateInComingCnt+d.gateOutGoingCnt) AS total
           FROM dailypassenger d JOIN stationinfo s ON d.staCode=s.staCode
           WHERE YEAR(d.trnOpDate) BETWEEN 2019 AND 2025
           GROUP BY s.staCode, s.staName, s.cluster_name, yr
           ORDER BY s.staCode, yr"""
    return pd.read_sql(q, con=engine)

@st.cache_data(ttl=300)
def load_station_forecast(sta_code):
    engine = create_engine(DB_URL)
    # 自動建表（forecasting.py 尚未執行時確保不崩潰）
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS stationforecast (
                staCode VARCHAR(10) NOT NULL,
                ds DATE NOT NULL,
                yhat FLOAT, yhat_lower FLOAT, yhat_upper FLOAT,
                is_future TINYINT(1) DEFAULT 0,
                PRIMARY KEY (staCode, ds)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """))
    try:
        q = text("SELECT ds, yhat, yhat_lower, yhat_upper, is_future "
                 "FROM stationforecast WHERE staCode=:code ORDER BY ds")
        df = pd.read_sql(q, con=engine, params={'code': sta_code})
    except Exception:
        return pd.DataFrame()
    if not df.empty:
        df['ds'] = pd.to_datetime(df['ds'])
    return df

def compute_cagr(yearly_df, base_year=2019, target_year=2024):
    rows = []
    for code, grp in yearly_df.groupby('staCode'):
        b = grp[grp['yr'] == base_year]['total'].values
        t = grp[grp['yr'] == target_year]['total'].values
        if len(b) == 0 or len(t) == 0 or b[0] == 0:
            continue
        n    = target_year - base_year
        cagr = (t[0] / b[0]) ** (1/n) - 1
        rows.append({
            'staCode': code,
            'staName': grp.iloc[0]['staName'],
            'cluster_name': grp.iloc[0]['cluster_name'],
            'base_total':   int(b[0]),
            'target_total': int(t[0]),
            'cagr_pct': round(cagr * 100, 2),
        })
    return pd.DataFrame(rows).sort_values('cagr_pct', ascending=False).reset_index(drop=True)

# ============================================================
# Sidebar
# ============================================================
# 造訪人數計數
# ============================================================
def record_visit():
    """首次載入時記錄一筆造訪，並回傳累計總次數。"""
    try:
        engine = create_engine(DB_URL)
        with engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS page_visits (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    visited_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """))
            conn.execute(text("INSERT INTO page_visits (visited_at) VALUES (NOW())"))
            result = conn.execute(text("SELECT COUNT(*) FROM page_visits"))
            return result.scalar()
    except Exception:
        return None

# 只在 session 第一次載入時計數，避免切換頁面重複計算
if 'visited' not in st.session_state:
    st.session_state['visited'] = True
    st.session_state['visit_count'] = record_visit()

# ============================================================
with st.sidebar:
    st.markdown("## 🚂 台鐵營運\n## 智慧分析系統")
    st.markdown("---")
    page = st.radio(
        "選擇分析模組",
        ["📊 營運主管戰情室",
         "🧬 車站 DNA 分群地圖",
         "🗺️ 全台車站地圖",
         "📈 時間序列趨勢分析",
         "🔮 運量預測（Phase 3）",
         "📉 CAGR 成長趨勢排行",
         "🚆 車種別客運分析",
         "📋 報告大綱與建議"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("資料範圍：2019/04 – 2025/12")
    st.caption("資料來源：台灣鐵路局")
    st.markdown("[🔗 資料來源連結](https://drive.google.com/drive/folders/12H-2c3i5waVKD_LnWpPWD2ORN8FJBkA8?usp=sharing)")
    if st.session_state.get('visit_count'):
        st.caption(f"👥 累計造訪：{st.session_state['visit_count']:,} 次")

# ============================================================
# 頁面一：營運主管戰情室
# ============================================================
if page == "📊 營運主管戰情室":
    st.title("📊 營運主管戰情室")
    st.markdown("整合全台 243 座車站 2019–2025 年度客流，提供關鍵營運指標一覽。")

    with st.expander("📖 分析原理說明", expanded=False):
        st.markdown("""
**本頁功能**：將所有分析模組的核心結果彙整成一個高階視圖，供管理層快速掌握全局。

#### KPI 指標計算方式
| 指標 | 計算邏輯 |
|------|---------|
| 累計總運量 | 加總 `gateInComingCnt + gateOutGoingCnt` 所有日期所有車站 |
| 最繁忙車站 | 依 `avg_daily_total`（歷史日均進出站人次）取最大值 |
| 運量月變化 | `(當月總量 − 上月總量) / 上月總量 × 100%` |
| 異常突波筆數 | 近 90 天內，各站 **Z-Score > 3** 的日子總計（見下方說明） |

#### 異常突波偵測（Z-Score）
Z-Score 是一種統計量，衡量某日運量偏離該站歷史均值的程度：

$$Z = \\frac{x - \\mu}{\\sigma}$$

- $x$：當日進出站人次
- $\\mu$：該站近期均值
- $\\sigma$：該站近期標準差
- **|Z| > 3** 表示當日運量極度異常（統計上約 0.3% 機率自然發生），視為突波警示。

#### 月度趨勢折線圖
橫軸為月份，縱軸為該月全台所有車站進出站人次加總。灰色陰影區標示 COVID-19 疫情對運量的衝擊區間（2020/03 – 2021/06）。
        """)

    cluster_df = load_cluster_data()
    monthly_df = load_monthly_total()

    col1, col2, col3, col4 = st.columns(4)
    total_pax   = int(monthly_df['total'].sum())
    top_station = cluster_df.iloc[0]['staName']
    top_daily   = int(cluster_df.iloc[0]['avg_daily_total'])
    latest = monthly_df.iloc[-1]; prev = monthly_df.iloc[-2]
    mom    = (latest['total'] - prev['total']) / prev['total'] * 100
    try:
        anom_count = len(load_recent_anomalies(90))
    except Exception:
        anom_count = 0

    col1.metric("累計總運量（2019–2025）", f"{total_pax/1e8:.2f} 億人次")
    col2.metric("最繁忙車站", top_station, f"日均 {fmt_number(top_daily)} 人")
    col3.metric("最新月份運量變化",
                latest['ym'].strftime('%Y-%m'),
                f"{'+' if mom>=0 else ''}{mom:.1f}% vs 上月")
    col4.metric("近 90 天異常突波", f"{anom_count} 筆")

    st.markdown("---")
    cl, cr = st.columns([3, 2])
    with cl:
        st.subheader("全台月度總運量趨勢（2019–2025）")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=monthly_df['ym'], y=monthly_df['total'],
                                 mode='lines', fill='tozeroy',
                                 line=dict(color='#1f77b4', width=2),
                                 fillcolor='rgba(31,119,180,0.15)'))
        fig.add_vrect(x0="2020-03-01", x1="2021-06-01",
                      fillcolor="rgba(255,0,0,0.07)", line_width=0,
                      annotation_text="COVID 衝擊", annotation_position="top left")
        fig.update_layout(xaxis_title="月份", yaxis_title="人次",
                          hovermode="x unified", height=350, margin=dict(t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)
    with cr:
        st.subheader("車站類型分布")
        cc = cluster_df['cluster_name'].value_counts().reset_index()
        cc.columns = ['cluster_name','count']
        fig2 = go.Figure(go.Pie(
            labels=cc['cluster_name'], values=cc['count'],
            marker_colors=[CLUSTER_COLORS.get(n,'#aaa') for n in cc['cluster_name']],
            hole=0.45, textinfo='label+percent'))
        fig2.update_layout(showlegend=False, height=350, margin=dict(t=10,b=10))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("⚠️ 近 90 天異常突波警示（Z-Score > 3）")
    try:
        adf = load_recent_anomalies(90)
        if len(adf):
            show = adf[['date','station','cluster_name','total','z']].copy()
            show['date']  = show['date'].dt.strftime('%Y-%m-%d')
            show['total'] = show['total'].apply(lambda x: f"{int(x):,}")
            show['z']     = show['z'].round(2)
            show.columns  = ['日期','車站','類型','進出人次','Z-Score']
            st.dataframe(show.head(30), use_container_width=True, hide_index=True)
        else:
            st.info("近 90 天無異常突波事件。")
    except Exception as e:
        st.warning(f"載入異常資料失敗：{e}")


# ============================================================
# 頁面二：車站 DNA 分群地圖
# ============================================================
elif page == "🧬 車站 DNA 分群地圖":
    st.title("🧬 車站 DNA 分群地圖")
    st.markdown("以 **K-Means (K=4)** 對全台車站聚類，用「日均規模」與「平假日行為」兩維度定義各站 DNA。")

    with st.expander("📖 分析原理說明", expanded=False):
        st.markdown("""
**K-Means 分群**是一種非監督式機器學習演算法，不需要事先標注資料，讓演算法自行找出資料中天然存在的群組。

#### 分析流程

**Step 1 — 特徵選取**：從 `v_StationFeatures` 視圖中，萃取每個車站兩個關鍵維度：
- `avg_daily_total`：歷史所有日期的平均每日進出站人次，代表車站的**絕對規模**
- `weekend_weekday_ratio`：假日（週六/日）日均 ÷ 平日（週一至五）日均，代表車站的**旅次屬性**（>1 偏觀光、<1 偏通勤）

**Step 2 — 標準化（StandardScaler）**：兩個特徵的數量級差異極大（人次動輒萬人，比值僅約 1.0），若直接計算距離，大數字會主導結果。因此用 Z-Score 標準化，使每個特徵的均值為 0、標準差為 1：

$$x' = \\frac{x - \\mu}{\\sigma}$$

**Step 3 — K-Means 訓練（K=4）**：演算法隨機初始化 4 個群心，反覆將每個車站分配給距離最近的群心，再重新計算群心，直到收斂。最終產生 4 個穩定的聚落。

**Step 4 — 自動語意命名**：根據各群心在原始空間中的位置（日均量排名 × 假日比高低）自動貼上人類可讀的標籤：

| 標籤 | 日均量 | 假日比 | 典型代表 |
|------|--------|--------|---------|
| 🏙️ 都會核心大站 | 最高 | — | 臺北 |
| 🚉 都會通勤大站 | 高 | 相對較低 | 桃園、臺中、板橋 |
| 🌊 週末觀光熱點 | — | 極高 | 枋野 |
| 🌾 在地一般站 | 低 | 中等 | 全台多數小站 |

**散佈圖解讀**：X 軸越右 = 車站越大；Y 軸越高 = 假日湧入越明顯。右下角為通勤型大站，左上角為觀光型小站。
        """)

    cluster_df = load_cluster_data()

    fig = px.scatter(
        cluster_df, x='avg_daily_total', y='weekend_weekday_ratio',
        color='cluster_name', color_discrete_map=CLUSTER_COLORS,
        hover_name='staName',
        hover_data={'avg_daily_total': ':,.0f', 'weekend_weekday_ratio': ':.3f',
                    'city': True, 'cluster_name': False},
        labels={'avg_daily_total': '日均進出站人次',
                'weekend_weekday_ratio': '假日 / 平日運量比',
                'cluster_name': '車站類型', 'city': '縣市'},
        title='全台車站 DNA 散佈圖', height=520,
    )
    fig.add_hline(y=1.0, line_dash='dot', line_color='gray',
                  annotation_text='假日＝平日 基準線')
    fig.update_traces(marker=dict(size=7, opacity=0.75))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("各群組詳細剖析")
    tabs = st.tabs([f"{CLUSTER_ICONS.get(n,'🔹')} {n}" for n in CLUSTER_COLORS])
    for tab, cname in zip(tabs, CLUSTER_COLORS):
        with tab:
            sub = cluster_df[cluster_df['cluster_name'] == cname]
            if sub.empty:
                st.info("請先執行 station_clustering.py"); continue
            c1,c2,c3 = st.columns(3)
            c1.metric("車站數", f"{len(sub)} 站")
            c2.metric("平均日運量", f"{int(sub['avg_daily_total'].mean()):,} 人")
            c3.metric("假日/平日比", f"{sub['weekend_weekday_ratio'].mean():.2f}")
            top20 = sub.head(20)
            fig_b = px.bar(top20, x='avg_daily_total', y='staName', orientation='h',
                           color_discrete_sequence=[CLUSTER_COLORS[cname]],
                           labels={'avg_daily_total':'日均人次','staName':''},
                           height=max(300, len(top20)*22))
            fig_b.update_layout(yaxis={'autorange':'reversed'}, margin=dict(t=10,b=10))
            st.plotly_chart(fig_b, use_container_width=True)
            with st.expander(f"查看全部 {len(sub)} 個車站"):
                show = sub[['staName','avg_daily_total','avg_weekend_total',
                            'avg_weekday_total','weekend_weekday_ratio','city']].copy()
                show.columns = ['車站','日均人次','假日均','平日均','假日/平日比','縣市']
                for c in ['日均人次','假日均','平日均']:
                    show[c] = show[c].apply(lambda x: f"{int(x):,}")
                st.dataframe(show, use_container_width=True, hide_index=True)


# ============================================================
# 頁面三：全台車站地圖
# ============================================================
elif page == "🗺️ 全台車站地圖":
    st.title("🗺️ 全台車站地圖")
    st.markdown("顏色 = 車站類型，大小 = 日均運量。滑鼠懸停可看詳細資訊。")

    with st.expander("📖 分析原理說明", expanded=False):
        st.markdown("""
**地理視覺化**將分群結果投影到真實地圖上，讓空間分布模式一目瞭然，彌補純數字表格無法呈現的地理洞察。

#### 資料來源
車站座標（經緯度）來自台鐵 API 提供的 `台鐵各車站詳細資料.json`，包含：
- `PositionLat` / `PositionLon`：WGS84 座標
- `LocationCity`：所屬縣市
- `StationClass`：車站等級（1 = 主要站、2 = 一般站 ...）

#### 圖層編碼邏輯
| 視覺屬性 | 對應資料 | 意義 |
|---------|---------|------|
| 圓點**顏色** | `cluster_name` 分群標籤 | 快速辨識車站營運類型 |
| 圓點**大小** | `avg_daily_total` 日均人次 | 面積正比於運量規模 |
| 底圖 | OpenStreetMap（免費、無需 Token） | 提供路網與地名參考 |

#### 如何解讀空間模式
- **紅點（都會核心）集中於北部平原**：反映台灣人口重心
- **橘點（觀光熱點）沿東部海岸線分布**：花蓮、台東地區的觀光需求
- **藍點（通勤大站）串連西部走廊**：桃園—台中—台南—高雄城際通勤帶
- **偏遠綠點密集的路段**：可能為服務稀疏人口的政策性路線，具社會功能但商業效益低
        """)

    cluster_df = load_cluster_data()
    geo = cluster_df.dropna(subset=['lat','lon'])

    # 篩選控制
    c1, c2 = st.columns([2,1])
    with c1:
        selected_clusters = st.multiselect(
            "篩選車站類型", list(CLUSTER_COLORS.keys()),
            default=list(CLUSTER_COLORS.keys())
        )
    with c2:
        min_vol = st.number_input("最低日均人次門檻", min_value=0,
                                  max_value=10000, value=0, step=500)

    filtered = geo[
        geo['cluster_name'].isin(selected_clusters) &
        (geo['avg_daily_total'] >= min_vol)
    ]

    fig_map = px.scatter_mapbox(
        filtered,
        lat='lat', lon='lon',
        color='cluster_name',
        color_discrete_map=CLUSTER_COLORS,
        size='avg_daily_total',
        size_max=30,
        hover_name='staName',
        hover_data={
            'city': True,
            'avg_daily_total': ':,.0f',
            'weekend_weekday_ratio': ':.2f',
            'cluster_name': False,
            'lat': False, 'lon': False,
        },
        labels={
            'avg_daily_total': '日均人次',
            'weekend_weekday_ratio': '假日/平日比',
            'cluster_name': '類型',
            'city': '縣市',
        },
        mapbox_style='open-street-map',
        zoom=7,
        center={'lat': 23.8, 'lon': 121.0},
        height=640,
    )
    fig_map.update_layout(margin=dict(t=0,b=0,l=0,r=0),
                          legend=dict(title='車站類型'))
    st.plotly_chart(fig_map, use_container_width=True)

    st.caption(f"顯示 {len(filtered)} / {len(geo)} 個車站")

    # 縣市分布
    st.markdown("---")
    st.subheader("各縣市車站運量排行")
    city_agg = filtered.groupby('city').agg(
        車站數=('staCode','count'),
        日均總人次=('avg_daily_total','sum'),
        最大站=('staName', lambda x: x.iloc[filtered.loc[x.index,'avg_daily_total'].argmax()])
    ).sort_values('日均總人次', ascending=False).reset_index()
    city_agg['日均總人次'] = city_agg['日均總人次'].apply(lambda x: f"{int(x):,}")
    city_agg.columns = ['縣市','車站數','日均總人次','最繁忙車站']
    st.dataframe(city_agg, use_container_width=True, hide_index=True)


# ============================================================
# 頁面四：時間序列趨勢分析（含多站比較 + 節慶標記）
# ============================================================
elif page == "📈 時間序列趨勢分析":
    st.title("📈 時間序列趨勢分析")

    with st.expander("📖 分析原理說明", expanded=False):
        st.markdown("""
本頁提供兩種模式，分別適合不同的分析情境。

#### 模式一：單站深度分析

**移動平均（Moving Average）**
每日客流受天氣、假期等短期因素干擾，波動劇烈。移動平均將前後 N 天取均值，平滑短期雜訊以還原長期趨勢：

$$MA_t = \\frac{1}{N} \\sum_{i=0}^{N-1} x_{t-i}$$

視窗 N 越大趨勢越平滑，但對轉折點的反應越遲鈍。預設 30 天為月度趨勢的良好折衷。

**異常突波偵測（滾動 Z-Score）**
以過去 N 天為基準，計算當日偏離程度：

$$Z_t = \\frac{x_t - \\bar{x}_{t-N:t}}{\\sigma_{t-N:t}}$$

- **Z > 3**（預設門檻）：表示當日運量高出近期均值超過 3 個標準差，在常態分布下發生機率 < 0.3%，視為統計異常。
- 多數異常日可對應至特定節慶（系統自動在表格中標示 ±3 天內的最近假日）。

**節慶垂直線**：由 `workalendar` 套件生成台灣法定節假日清單，疊加在圖表上輔助人工解讀。

---

#### 模式二：多站比較

同時繪製 2–5 個車站的移動平均趨勢線於同一圖表，用於：
- **競爭路線分析**（如桃園 vs 中壢）
- **城際對比**（臺北 vs 臺南）
- **觀光 vs 通勤**屬性的行為差異比較

各站使用不同顏色區分，X 軸對齊，可直接觀察各站的同步漲跌與分化趨勢。
        """)

    cluster_df   = load_cluster_data()
    station_list = load_station_list()
    holidays_df  = get_taiwan_holidays()

    options = {f"{r['staName']}（{r['staCode']}）": r['staCode']
               for _, r in station_list.iterrows()}

    # ── 模式切換 ────────────────────────────────────────────
    mode = st.radio("分析模式", ["單站深度分析", "多站比較"], horizontal=True)

    if mode == "多站比較":
        # 多站比較
        st.markdown("---")
        default_labels = [k for k in options if '桃園' in k or '瑞芳' in k or '臺南' in k][:3]
        selected_labels = st.multiselect("選擇 2–5 個車站", list(options.keys()),
                                         default=default_labels[:3], max_selections=5)
        if len(selected_labels) < 2:
            st.info("請至少選擇 2 個車站。"); st.stop()

        codes = tuple(options[l] for l in selected_labels)
        ma_w  = st.slider("移動平均天數", 7, 60, 30)

        with st.spinner("載入資料中..."):
            multi_df = load_multi_station_timeseries(codes)

        fig_m = go.Figure()
        for sta in multi_df['station'].unique():
            sub = multi_df[multi_df['station'] == sta].set_index('date').sort_index()
            sub['ma'] = sub['total_passengers'].rolling(ma_w, min_periods=ma_w//2).mean()
            fig_m.add_trace(go.Scatter(
                x=sub.index, y=sub['ma'], mode='lines', name=sta,
                hovertemplate='%{x|%Y-%m-%d}<br>MA：%{y:,.0f}<extra>' + sta + '</extra>',
            ))

        # 加入節慶垂直線（僅標記重要假日）
        show_holidays = st.checkbox("顯示重大節假日標記", value=True)
        if show_holidays:
            major = ['農曆除夕', '農曆新年', '清明節', '端午節', '中秋節', '國慶日']
            mh = holidays_df[holidays_df['holiday'].isin(major)]
            for _, hr in mh.iterrows():
                if hr['ds'] >= pd.Timestamp('2019-01-01'):
                    fig_m.add_vline(x=hr['ds'].timestamp()*1000,
                                    line_dash='dot', line_color='rgba(200,0,0,0.3)',
                                    annotation_text=hr['holiday'][:3] if hr['ds'].month in [1,2] else '')

        fig_m.update_layout(
            title="多站運量趨勢比較（移動平均）",
            xaxis_title="日期", yaxis_title="進出站人次（移動平均）",
            hovermode="x unified", height=480,
            legend=dict(orientation='h', yanchor='bottom', y=1.02),
        )
        st.plotly_chart(fig_m, use_container_width=True)

        # 平均運量比較表
        st.markdown("---")
        st.subheader("各站統計摘要")
        rows = []
        for sta in multi_df['station'].unique():
            sub = multi_df[multi_df['station'] == sta]
            rows.append({'車站': sta,
                         '日均人次': f"{int(sub['total_passengers'].mean()):,}",
                         '歷史峰值': f"{int(sub['total_passengers'].max()):,}",
                         '峰值日期': sub.loc[sub['total_passengers'].idxmax(), 'date'].strftime('%Y-%m-%d')})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    else:
        # 單站深度分析
        st.markdown("---")
        c_sel, c_ma, c_th = st.columns([3,1,1])
        with c_sel:
            default_idx = next((i for i,k in enumerate(options) if '桃園' in k), 0)
            sel = st.selectbox("選擇車站", list(options.keys()), index=default_idx)
        with c_ma:
            ma_w = st.number_input("移動平均天數", 7, 90, 30, step=7)
        with c_th:
            z_th = st.number_input("異常 Z-Score 門檻", 1.0, 5.0, 3.0, step=0.5)

        code  = options[sel]
        sname = sel.split('（')[0]
        sta_info = cluster_df[cluster_df['staCode'] == code]
        if not sta_info.empty:
            cname = sta_info.iloc[0]['cluster_name']
            st.markdown(
                f"<span style='background:{CLUSTER_COLORS.get(cname,'#666')};"
                f"color:white;padding:3px 10px;border-radius:12px;font-size:0.85rem'>"
                f"{CLUSTER_ICONS.get(cname,'')} {cname}</span>",
                unsafe_allow_html=True,
            )

        with st.spinner(f"載入 {sname} 資料中..."):
            ts = load_station_timeseries(code).set_index('date').sort_index()

        if ts.empty:
            st.warning("此車站無資料"); st.stop()

        ts['ma'] = ts['total_passengers'].rolling(ma_w, min_periods=ma_w//2).mean()
        anom_mask, z_scores = anomaly_detect(ts['total_passengers'], ma_w, z_th)
        ts['z_score']    = z_scores
        ts['is_anomaly'] = anom_mask
        anomalies = ts[ts['is_anomaly']]

        # KPI
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("資料期間", f"{ts.index.min().year}–{ts.index.max().year}")
        c2.metric("日均運量", f"{int(ts['total_passengers'].mean()):,} 人")
        c3.metric("歷史峰值", f"{int(ts['total_passengers'].max()):,} 人",
                  ts['total_passengers'].idxmax().strftime('%Y-%m-%d'))
        c4.metric(f"異常突波（Z>{z_th}）", f"{len(anomalies)} 天")

        lc = CLUSTER_COLORS.get(sta_info.iloc[0]['cluster_name'] if not sta_info.empty else '', '#1f77b4')
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(
            x=ts.index, y=ts['total_passengers'], mode='lines', name='每日實際',
            line=dict(color='rgba(180,180,180,0.5)', width=1)))
        fig_ts.add_trace(go.Scatter(
            x=ts.index, y=ts['ma'], mode='lines', name=f'{ma_w}天移動平均',
            line=dict(color=lc, width=2.5)))
        if not anomalies.empty:
            fig_ts.add_trace(go.Scatter(
                x=anomalies.index, y=anomalies['total_passengers'],
                mode='markers', name=f'異常（Z>{z_th}）',
                marker=dict(color='red', size=8),
                customdata=anomalies['z_score'],
                hovertemplate='%{x|%Y-%m-%d}<br>人次：%{y:,}<br>Z：%{customdata:.2f}<extra></extra>'))

        # 節慶垂直線
        show_hol = st.checkbox("顯示重大節假日標記", value=True)
        if show_hol:
            major = ['農曆除夕', '農曆新年', '清明節', '端午節', '中秋節', '國慶日']
            mh = holidays_df[holidays_df['holiday'].isin(major)]
            for _, hr in mh.iterrows():
                if hr['ds'] >= ts.index.min() and hr['ds'] <= ts.index.max():
                    fig_ts.add_vline(
                        x=hr['ds'].timestamp()*1000,
                        line_dash='dot', line_color='rgba(200,100,0,0.35)',
                        annotation_text=hr['holiday'][:2],
                        annotation_font_size=9,
                    )

        fig_ts.update_layout(
            title=f"{sname} 客流趨勢與異常偵測",
            xaxis=dict(title="日期", rangeslider=dict(visible=True), type='date'),
            yaxis_title="進出站人次",
            hovermode="x unified", height=500,
            legend=dict(orientation='h', yanchor='bottom', y=1.02),
        )
        st.plotly_chart(fig_ts, use_container_width=True)

        # 月均季節性
        ts_r = ts.reset_index()
        monthly_avg = ts_r.groupby(ts_r['date'].dt.month)['total_passengers'].mean().reset_index()
        monthly_avg.columns = ['month','avg']
        monthly_avg['月份'] = monthly_avg['month'].apply(lambda m: f"{m} 月")
        fig_mon = px.bar(monthly_avg, x='月份', y='avg',
                         color_discrete_sequence=[lc],
                         labels={'avg':'月均人次'},
                         title='月均分佈（季節性）', height=280)
        fig_mon.update_layout(margin=dict(t=40,b=10))
        st.plotly_chart(fig_mon, use_container_width=True)

        # 異常事件與節慶對照
        if not anomalies.empty:
            st.markdown("---")
            st.subheader(f"⚠️ {len(anomalies)} 筆異常突波")
            anom_show = anomalies[['total_passengers','z_score']].copy().reset_index()
            anom_show.columns = ['日期','人次','Z-Score']
            # 比對最近節假日（±3天內）
            def nearest_holiday(date):
                diff = (holidays_df['ds'] - date).abs()
                idx  = diff.idxmin()
                if diff[idx].days <= 3:
                    return holidays_df.loc[idx, 'holiday']
                return ''
            anom_show['最近節慶'] = anom_show['日期'].apply(nearest_holiday)
            anom_show['日期']  = anom_show['日期'].dt.strftime('%Y-%m-%d')
            anom_show['人次']  = anom_show['人次'].apply(lambda x: f"{int(x):,}")
            anom_show['Z-Score'] = anom_show['Z-Score'].round(2)
            st.dataframe(anom_show.sort_values('Z-Score', ascending=False),
                         use_container_width=True, hide_index=True)


# ============================================================
# 頁面五：運量預測（Phase 3）
# ============================================================
elif page == "🔮 運量預測（Phase 3）":
    st.title("🔮 運量預測（Phase 3）")
    st.markdown(
        "使用 **Facebook Prophet** 模型，融合年/週季節性與台灣節假日，"
        "預測未來 **90 天** 各站運量及 90% 信賴區間。"
    )

    with st.expander("📖 分析原理說明", expanded=False):
        st.markdown("""
**Prophet** 是 Meta（Facebook）開源的時間序列預測框架，專為有明顯季節性、節假日效應，且資料中存在缺漏或異常值的商業資料設計。

#### 模型分解架構

Prophet 將時間序列拆解為三個可加法疊加的成分：

$$y(t) = g(t) + s(t) + h(t) + \\epsilon_t$$

| 成分 | 符號 | 說明 |
|------|------|------|
| 趨勢 | $g(t)$ | 長期線性或邏輯成長趨勢，自動偵測趨勢轉折點（changepoints） |
| 季節性 | $s(t)$ | 週期性規律，本系統啟用**年度季節性**（春節、暑假）與**週度季節性**（週末效應） |
| 節假日 | $h(t)$ | 特定日期的脈衝效應，輸入台灣法定假日清單後，模型學習每個節慶的拉抬幅度 |
| 誤差 | $\\epsilon_t$ | 不可預測的隨機雜訊 |

#### 本系統設定
- `changepoint_prior_scale = 0.05`：趨勢彈性設為保守，避免過度擬合短期突波
- `interval_width = 0.9`：輸出 **90% 信賴區間**，灰藍色帶表示模型對預測的不確定程度
- 節假日資料：由 `workalendar` 生成 2019–2026 年台灣所有法定假日

#### 信賴區間解讀
- **預測線（虛線）**：模型的中位數預測值
- **藍色信賴帶**：有 90% 的把握，真實值會落在此範圍內
- 信賴帶越寬，代表模型對該時段的不確定性越高（如連假、特殊事件前後）

#### 快取機制
執行 `python forecasting.py` 可預先批次計算全台 242 站的預測結果存入資料庫，之後開啟此頁面即可秒速載入，無需等待。
        """)

    station_list = load_station_list()
    cluster_df   = load_cluster_data()
    holidays_df  = get_taiwan_holidays()
    options = {f"{r['staName']}（{r['staCode']}）": r['staCode']
               for _, r in station_list.iterrows()}

    c_sel, c_days = st.columns([3,1])
    with c_sel:
        default_idx = next((i for i,k in enumerate(options) if '桃園' in k), 0)
        sel  = st.selectbox("選擇車站", list(options.keys()), index=default_idx)
    with c_days:
        forecast_days = st.number_input("預測天數", 30, 180, 90, step=30)

    code  = options[sel]
    sname = sel.split('（')[0]
    sta_info = cluster_df[cluster_df['staCode'] == code]

    st.markdown("---")

    # 先嘗試讀取預先計算的結果
    cached_fc = load_station_forecast(code)

    if not cached_fc.empty:
        st.success(f"已從資料庫載入 {sname} 的預測結果（執行 forecasting.py 更新）")
        hist_df = load_station_timeseries(code).rename(columns={'date':'ds','total_passengers':'y'})
        fc = cached_fc.rename(columns={'ds':'ds'})

        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(
            x=hist_df['ds'], y=hist_df['y'], mode='lines', name='歷史實際',
            line=dict(color='rgba(150,150,150,0.4)', width=1)))
        hist_fc = fc[fc['is_future'] == 0]
        fut_fc  = fc[fc['is_future'] == 1].tail(forecast_days)
        # 信賴區間
        fig_fc.add_trace(go.Scatter(
            x=pd.concat([fut_fc['ds'], fut_fc['ds'][::-1]]),
            y=pd.concat([fut_fc['yhat_upper'], fut_fc['yhat_lower'][::-1]]),
            fill='toself', fillcolor='rgba(31,119,180,0.15)',
            line=dict(color='rgba(0,0,0,0)'), name='90% 信賴區間'))
        fig_fc.add_trace(go.Scatter(
            x=fut_fc['ds'], y=fut_fc['yhat'], mode='lines', name='預測值',
            line=dict(color='#1f77b4', width=2.5, dash='dot')))
        fig_fc.add_vline(x=hist_df['ds'].max().timestamp()*1000,
                         line_dash='dash', line_color='gray',
                         annotation_text='預測起點')
        fig_fc.update_layout(
            title=f"{sname} 未來 {forecast_days} 天運量預測",
            xaxis_title="日期", yaxis_title="預估進出站人次",
            hovermode="x unified", height=480,
            legend=dict(orientation='h', yanchor='bottom', y=1.02))
        st.plotly_chart(fig_fc, use_container_width=True)

        # 預測摘要
        st.subheader("未來 30 / 60 / 90 天預測摘要")
        for d in [30, 60, min(90, forecast_days)]:
            sub = fut_fc.head(d)
            if sub.empty: continue
            c1,c2,c3 = st.columns(3)
            c1.metric(f"未來 {d} 天日均（預測）", f"{int(sub['yhat'].mean()):,} 人")
            c2.metric(f"最高峰（預測）",
                      f"{int(sub['yhat'].max()):,} 人",
                      sub.loc[sub['yhat'].idxmax(),'ds'].strftime('%Y-%m-%d'))
            c3.metric(f"90% 區間寬度",
                      f"± {int((sub['yhat_upper']-sub['yhat_lower']).mean()//2):,} 人")

    else:
        # 即時計算
        st.info(f"資料庫尚無 {sname} 的預測快取。點擊下方按鈕即時計算（約 30–60 秒）。")
        st.caption("💡 可在背景執行 `python forecasting.py` 預先算好全部車站，之後秒開。")

        if st.button(f"🔮 立即預測 {sname} 未來 {forecast_days} 天", type="primary"):
            try:
                from prophet import Prophet
            except ImportError:
                st.error("Prophet 套件未安裝。請在本機執行預測，或先執行 forecasting.py 將結果存入資料庫後，雲端即可讀取快取結果。")
                st.stop()

            with st.spinner(f"Prophet 訓練中，請稍候..."):
                ts = load_station_timeseries(code)
                prophet_df = ts.rename(columns={'date':'ds','total_passengers':'y'})
                model = Prophet(
                    holidays=holidays_df[['ds','holiday']],
                    yearly_seasonality=True, weekly_seasonality=True,
                    daily_seasonality=False,
                    changepoint_prior_scale=0.05,
                    interval_width=0.9,
                )
                model.fit(prophet_df)
                future   = model.make_future_dataframe(periods=forecast_days)
                forecast = model.predict(future)

            # 繪圖
            fut = forecast[forecast['ds'] > prophet_df['ds'].max()]
            fig_fc = go.Figure()
            fig_fc.add_trace(go.Scatter(
                x=prophet_df['ds'], y=prophet_df['y'], mode='lines', name='歷史實際',
                line=dict(color='rgba(150,150,150,0.4)', width=1)))
            fig_fc.add_trace(go.Scatter(
                x=pd.concat([fut['ds'], fut['ds'][::-1]]),
                y=pd.concat([fut['yhat_upper'], fut['yhat_lower'][::-1]]),
                fill='toself', fillcolor='rgba(31,119,180,0.15)',
                line=dict(color='rgba(0,0,0,0)'), name='90% 信賴區間'))
            fig_fc.add_trace(go.Scatter(
                x=fut['ds'], y=fut['yhat'], mode='lines', name='預測值',
                line=dict(color='#1f77b4', width=2.5, dash='dot')))
            fig_fc.add_vline(x=prophet_df['ds'].max().timestamp()*1000,
                             line_dash='dash', line_color='gray',
                             annotation_text='預測起點')
            fig_fc.update_layout(
                title=f"{sname} 未來 {forecast_days} 天運量預測",
                xaxis_title="日期", yaxis_title="預估進出站人次",
                hovermode="x unified", height=480,
                legend=dict(orientation='h', yanchor='bottom', y=1.02))
            st.plotly_chart(fig_fc, use_container_width=True)

            # 摘要
            for d in [30, 60, min(90, forecast_days)]:
                sub = fut.head(d)
                if sub.empty: continue
                c1,c2,c3 = st.columns(3)
                c1.metric(f"未來 {d} 天日均", f"{int(sub['yhat'].mean()):,} 人")
                c2.metric("最高峰（預測）", f"{int(sub['yhat'].max()):,} 人",
                          sub.loc[sub['yhat'].idxmax(),'ds'].strftime('%Y-%m-%d'))
                c3.metric("90% 區間寬度",
                          f"± {int((sub['yhat_upper']-sub['yhat_lower']).mean()//2):,} 人")

            st.caption("此結果未快取，下次選擇同一站會重新計算。"
                       "建議執行 forecasting.py 將全部車站存入資料庫。")


# ============================================================
# 頁面六：CAGR 成長趨勢排行
# ============================================================
elif page == "📉 CAGR 成長趨勢排行":
    st.title("📉 CAGR 成長趨勢排行")
    st.markdown(
        "計算各站 **2019→2024** 的年複合成長率（CAGR），"
        "找出「新興崛起」與「逐漸衰退」的車站。"
    )

    with st.expander("📖 分析原理說明", expanded=False):
        st.markdown("""
**CAGR（Compound Annual Growth Rate，年複合成長率）** 是衡量一個數值在多年間平均每年成長多少百分比的指標，排除單一年份的劇烈波動，反映長期結構性趨勢。

#### 計算公式

$$CAGR = \\left(\\frac{\\text{目標年總運量}}{\\text{基準年總運量}}\\right)^{\\frac{1}{n}} - 1$$

其中 $n$ 為年數（例如 2019→2024 則 $n=5$）。

**範例**：若某站 2019 年運量 100 萬人次，2024 年為 160 萬人次：

$$CAGR = \\left(\\frac{160}{100}\\right)^{\\frac{1}{5}} - 1 \\approx 9.9\\%$$

代表該站平均每年約成長 9.9%。

#### 為何選擇 CAGR 而非單純比較增減？

| 方法 | 缺點 |
|------|------|
| 直接相減（+60萬） | 無法跨站比較（大站基數大，數字自然高）|
| 成長百分比（+60%） | 忽略時間長度，5年+60% 與 2年+60% 意義不同 |
| **CAGR** | 標準化為年化率，可跨站、跨期公平比較 ✅ |

#### 注意事項
- **COVID 效應**：2020–2021 年運量大幅萎縮，若選 2020 為基準年，CAGR 會偏高（因基數低）；選 2019 為基準可反映疫情前後的真實結構變化。
- **2025 年資料**：僅有部分月份，選為目標年時數字會偏低，建議選 2024。
- **負 CAGR** 表示該站長期衰退，可能原因包括人口外移、競爭運具（高鐵）替代、或班次縮減。
        """)


    c1, c2 = st.columns([2,2])
    with c1:
        base_yr   = st.selectbox("基準年", [2019, 2020, 2021], index=0)
    with c2:
        target_yr = st.selectbox("比較年", [2023, 2024, 2025], index=1)

    with st.spinner("計算 CAGR 中..."):
        yearly_df = load_yearly_totals()
        cagr_df   = compute_cagr(yearly_df, base_yr, target_yr)

    if cagr_df.empty:
        st.warning("所選年份資料不足。"); st.stop()

    st.markdown("---")
    ct, cb = st.columns(2)
    with ct:
        st.subheader(f"🚀 成長最快 TOP 20（{base_yr}→{target_yr}）")
        top20 = cagr_df.head(20)
        fig_t = px.bar(
            top20, x='cagr_pct', y='staName', orientation='h',
            color='cluster_name', color_discrete_map=CLUSTER_COLORS,
            labels={'cagr_pct':'CAGR (%)','staName':'','cluster_name':'類型'},
            height=550,
        )
        fig_t.update_layout(yaxis={'autorange':'reversed'}, margin=dict(t=10,b=10))
        st.plotly_chart(fig_t, use_container_width=True)

    with cb:
        st.subheader(f"📉 衰退最深 TOP 20（{base_yr}→{target_yr}）")
        bot20 = cagr_df.tail(20).iloc[::-1]
        fig_b = px.bar(
            bot20, x='cagr_pct', y='staName', orientation='h',
            color='cluster_name', color_discrete_map=CLUSTER_COLORS,
            labels={'cagr_pct':'CAGR (%)','staName':'','cluster_name':'類型'},
            height=550,
        )
        fig_b.update_layout(yaxis={'autorange':'reversed'}, margin=dict(t=10,b=10))
        st.plotly_chart(fig_b, use_container_width=True)

    st.markdown("---")
    st.subheader("逐年運量趨勢（選取車站）")
    trend_opts  = {r['staName']: r['staCode'] for _, r in
                   pd.read_sql("SELECT staCode,staName FROM stationinfo",
                               create_engine(DB_URL)).iterrows()}
    sel_stations = st.multiselect(
        "選擇車站追蹤年度趨勢", list(trend_opts.keys()),
        default=list(trend_opts.keys())[:3]
    )
    if sel_stations:
        sel_codes = [trend_opts[s] for s in sel_stations]
        sub_yr = yearly_df[yearly_df['staCode'].isin(sel_codes)]
        fig_yr = px.line(
            sub_yr, x='yr', y='total', color='staName',
            markers=True,
            labels={'yr':'年份','total':'年總人次','staName':'車站'},
            title='各站年度總運量趨勢', height=380,
        )
        fig_yr.update_layout(xaxis=dict(tickmode='linear', tick0=2019, dtick=1))
        st.plotly_chart(fig_yr, use_container_width=True)

    st.markdown("---")
    st.subheader("完整 CAGR 排行表")
    show_cagr = cagr_df[['staName','cluster_name','base_total','target_total','cagr_pct']].copy()
    show_cagr.columns = ['車站','類型',f'{base_yr}年總人次',f'{target_yr}年總人次','CAGR (%)']
    for c in [f'{base_yr}年總人次', f'{target_yr}年總人次']:
        show_cagr[c] = show_cagr[c].apply(lambda x: f"{x:,}")
    st.dataframe(show_cagr, use_container_width=True, hide_index=True)


# ============================================================
# 頁面七：車種別客運分析
# ============================================================
elif page == "🚆 車種別客運分析":
    st.title("🚆 車種別客運分析")
    st.markdown("以 **自強號、莒光號、區間列車、普通車** 四車種切入，分析 2019–2025 年度客運結構演變。")

    with st.expander("📖 分析原理說明", expanded=False):
        st.markdown("""
**本頁資料來源**：台灣鐵路局公開統計「各年月車種別旅客人數及延人公里」，涵蓋 2019/01 – 2025/12，共四車種：

| 車種 | 說明 |
|------|------|
| 自強號 | 長途對號快車，票價最高，平均旅程最長 |
| 莒光號 | 中長途對號列車，近年因老化大幅縮減班次 |
| 區間列車 | 通勤短途為主，班次最密，客運量占比最大 |
| 普通車 | 最慢速，2020 年後幾乎停駛，資料極少 |

#### 各分析說明

**1. COVID-19 衝擊**：以 2019 年為基準，計算各年度總客運量變化百分比，量化疫情對各車種的實際衝擊程度。

**2. 車種市占率**：堆疊面積圖顯示各車種每月人次佔總人次的比例，反映旅客搭乘行為結構是否改變（如疫後旅客是否更傾向自強號長途旅遊）。

**3. 季節性模式**：對 2019–2025 各月取平均值，排除年際差異後觀察純粹的月份效應。高峰月份通常為 1–2 月（春節）、7–8 月（暑假）。

**4. 平均旅程距離**：計算公式為：

$$\\text{平均旅程（km）} = \\frac{\\text{延人公里}}{\\text{旅客人次}}$$

自強號旅程最長（長途），區間列車最短（通勤），可觀察是否因高鐵競爭導致台鐵長途旅客逐年轉移。

**5. CAGR（複合年增長率）**：

$$\\text{CAGR} = \\left(\\frac{\\text{目標年人次}}{\\text{基準年人次}}\\right)^{\\frac{1}{n}} - 1$$

其中 $n$ 為年數差。CAGR > 0 表示成長，< 0 表示衰退，可客觀比較各車種的長期發展趨勢。
        """)

    tt = load_train_type_data()
    TRAIN_COLORS = {
        '自強號':  '#d62728',
        '莒光號':  '#ff7f0e',
        '區間列車': '#1f77b4',
        '普通車':  '#2ca02c',
    }

    # ── Section 1：COVID-19 衝擊 ──────────────────────────────
    st.markdown("---")
    st.subheader("1. COVID-19 衝擊分析")

    fig_covid = go.Figure()
    fig_covid.add_trace(go.Scatter(
        x=tt['ym'], y=tt['pax_total'], mode='lines', fill='tozeroy',
        name='總客運人次', line=dict(color='#1f77b4', width=2),
        fillcolor='rgba(31,119,180,0.12)'
    ))
    for col, name in [('pax_tzechiang','自強號'),('pax_chukuang','莒光號'),
                      ('pax_local','區間列車'),('pax_ordinary','普通車')]:
        fig_covid.add_trace(go.Scatter(
            x=tt['ym'], y=tt[col], mode='lines', name=name,
            line=dict(color=TRAIN_COLORS[name], width=1.5, dash='dot')
        ))
    fig_covid.add_vrect(x0='2020-02-01', x1='2020-05-01',
        fillcolor='rgba(255,0,0,0.08)', line_width=0,
        annotation_text='COVID 第一波', annotation_position='top left')
    fig_covid.add_vrect(x0='2021-05-01', x1='2021-08-01',
        fillcolor='rgba(255,0,0,0.08)', line_width=0,
        annotation_text='COVID 第三級', annotation_position='top left')
    fig_covid.update_layout(
        xaxis_title='月份', yaxis_title='人次',
        hovermode='x unified', height=380, margin=dict(t=10, b=10)
    )
    st.plotly_chart(fig_covid, use_container_width=True)

    # 衝擊量化表
    yearly_tt = tt.groupby('year')[['pax_total','pax_tzechiang','pax_chukuang','pax_local']].sum()
    base_2019 = yearly_tt.loc[2019]
    impact = pd.DataFrame({
        '年份': yearly_tt.index,
        '總人次': yearly_tt['pax_total'],
        'vs 2019 (%)': ((yearly_tt['pax_total'] / base_2019['pax_total']) - 1) * 100,
    })
    impact['總人次'] = impact['總人次'].apply(lambda x: f"{x:,.0f}")
    impact['vs 2019 (%)'] = impact['vs 2019 (%)'].apply(lambda x: f"{x:+.1f}%")
    st.dataframe(impact.set_index('年份'), use_container_width=True)

    # ── Section 2：車種結構變遷 ───────────────────────────────
    st.markdown("---")
    st.subheader("2. 車種市占率結構變遷")

    tt['share_tzechiang'] = tt['pax_tzechiang'] / tt['pax_total'] * 100
    tt['share_chukuang']  = tt['pax_chukuang']  / tt['pax_total'] * 100
    tt['share_local']     = tt['pax_local']      / tt['pax_total'] * 100
    tt['share_ordinary']  = tt['pax_ordinary']   / tt['pax_total'] * 100

    fig_share = go.Figure()
    for col, name in [('share_tzechiang','自強號'),('share_chukuang','莒光號'),
                      ('share_local','區間列車'),('share_ordinary','普通車')]:
        fig_share.add_trace(go.Scatter(
            x=tt['ym'], y=tt[col], mode='lines', name=name,
            stackgroup='one', line=dict(color=TRAIN_COLORS[name])
        ))
    fig_share.update_layout(
        xaxis_title='月份', yaxis_title='市占率 (%)',
        hovermode='x unified', height=360, margin=dict(t=10, b=10),
        yaxis=dict(range=[0, 100])
    )
    st.plotly_chart(fig_share, use_container_width=True)

    # 年均市占率比較
    yr_share = tt.groupby('year').apply(lambda g: pd.Series({
        '自強號':   g['pax_tzechiang'].sum() / g['pax_total'].sum() * 100,
        '莒光號':   g['pax_chukuang'].sum()  / g['pax_total'].sum() * 100,
        '區間列車': g['pax_local'].sum()      / g['pax_total'].sum() * 100,
        '普通車':   g['pax_ordinary'].sum()   / g['pax_total'].sum() * 100,
    })).round(1)
    st.dataframe(yr_share.style.format("{:.1f}%"), use_container_width=True)

    # ── Section 3：季節性模式 ─────────────────────────────────
    st.markdown("---")
    st.subheader("3. 季節性模式（月均人次）")

    monthly_avg = tt.groupby('month')[['pax_tzechiang','pax_chukuang',
                                        'pax_local','pax_ordinary']].mean().reset_index()
    month_labels = ['1月','2月','3月','4月','5月','6月',
                    '7月','8月','9月','10月','11月','12月']
    monthly_avg['month_label'] = monthly_avg['month'].apply(lambda m: month_labels[m-1])

    fig_season = go.Figure()
    for col, name in [('pax_tzechiang','自強號'),('pax_chukuang','莒光號'),
                      ('pax_local','區間列車'),('pax_ordinary','普通車')]:
        fig_season.add_trace(go.Bar(
            x=monthly_avg['month_label'], y=monthly_avg[col],
            name=name, marker_color=TRAIN_COLORS[name]
        ))
    fig_season.update_layout(
        barmode='stack', xaxis_title='月份', yaxis_title='月均人次',
        hovermode='x unified', height=360, margin=dict(t=10, b=10)
    )
    st.plotly_chart(fig_season, use_container_width=True)

    # ── Section 4：平均旅程距離 ───────────────────────────────
    st.markdown("---")
    st.subheader("4. 平均旅程距離（延人公里 ÷ 人次）")

    km_df = tt[tt['km_total'] > 0].copy()
    km_df['avg_dist_total']     = km_df['km_total']     / km_df['pax_total']
    km_df['avg_dist_tzechiang'] = km_df['km_tzechiang'] / km_df['pax_tzechiang'].replace(0, np.nan)
    km_df['avg_dist_chukuang']  = km_df['km_chukuang']  / km_df['pax_chukuang'].replace(0, np.nan)
    km_df['avg_dist_local']     = km_df['km_local']      / km_df['pax_local'].replace(0, np.nan)

    fig_dist = go.Figure()
    for col, name in [('avg_dist_tzechiang','自強號'),('avg_dist_chukuang','莒光號'),
                      ('avg_dist_local','區間列車')]:
        fig_dist.add_trace(go.Scatter(
            x=km_df['ym'], y=km_df[col], mode='lines', name=name,
            line=dict(color=TRAIN_COLORS[name], width=2)
        ))
    fig_dist.update_layout(
        xaxis_title='月份', yaxis_title='平均旅程（公里）',
        hovermode='x unified', height=360, margin=dict(t=10, b=10)
    )
    st.plotly_chart(fig_dist, use_container_width=True)
    st.caption("普通車因資料缺漏較多，不納入比較。")

    # ── Section 5：車種 CAGR ──────────────────────────────────
    st.markdown("---")
    st.subheader("5. 各車種年度成長率（CAGR）")

    yr_pax = tt.groupby('year')[['pax_tzechiang','pax_chukuang','pax_local']].sum()
    available_years = sorted(yr_pax.index.tolist())
    c1, c2 = st.columns(2)
    base_y   = c1.selectbox("基準年", available_years, index=0)
    target_y = c2.selectbox("目標年", available_years, index=len(available_years)-1)

    if base_y < target_y:
        n = target_y - base_y
        cagr_rows = []
        for col, name in [('pax_tzechiang','自強號'),('pax_chukuang','莒光號'),('pax_local','區間列車')]:
            b = yr_pax.loc[base_y, col]
            t = yr_pax.loc[target_y, col]
            if b > 0:
                cagr_rows.append({'車種': name, f'{base_y}年': int(b),
                                   f'{target_y}年': int(t),
                                   'CAGR (%)': round(((t/b)**(1/n)-1)*100, 2)})
        cagr_tt = pd.DataFrame(cagr_rows)

        fig_cagr = px.bar(
            cagr_tt, x='車種', y='CAGR (%)',
            color='車種', color_discrete_map=TRAIN_COLORS,
            text='CAGR (%)', height=340
        )
        fig_cagr.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig_cagr.update_layout(showlegend=False, margin=dict(t=20, b=10))
        st.plotly_chart(fig_cagr, use_container_width=True)

        cagr_tt[f'{base_y}年'] = cagr_tt[f'{base_y}年'].apply(lambda x: f"{x:,}")
        cagr_tt[f'{target_y}年'] = cagr_tt[f'{target_y}年'].apply(lambda x: f"{x:,}")
        st.dataframe(cagr_tt.set_index('車種'), use_container_width=True)
    else:
        st.warning("目標年須大於基準年")

# ============================================================
# 頁面八：報告大綱與建議
# ============================================================
elif page == "📋 報告大綱與建議":
    st.title("📋 台鐵營運分析報告大綱與建議")
    st.markdown("根據本系統各分析模組的資料，自動彙整報告架構，並提供初步策略建議供參考。")

    with st.expander("📖 本頁功能說明", expanded=False):
        st.markdown("""
本頁自動讀取各分析模組的最新計算結果，整合成一份**報告大綱**，協助使用者：
- 快速掌握各章節應涵蓋的核心發現
- 了解資料所呈現的趨勢與異常
- 取得初步的策略建議作為報告撰寫起點

> 建議搭配各分析頁面的圖表，將大綱內容具體化為完整報告。
        """)

    # ── 讀取各模組資料 ────────────────────────────────────────
    try:
        monthly_df   = load_monthly_total()
        cluster_df   = load_cluster_data()
        yearly_df    = load_yearly_totals()
        cagr_df      = compute_cagr(yearly_df, base_year=2019, target_year=2024)
        tt           = load_train_type_data()
        anom_df      = load_recent_anomalies(90)
        data_ok      = True
    except Exception as e:
        st.error(f"資料載入失敗：{e}")
        data_ok = False

    if data_ok:
        # ── 計算摘要數字 ──────────────────────────────────────
        total_pax      = int(monthly_df['total'].sum())
        latest_month   = monthly_df.iloc[-1]
        prev_month     = monthly_df.iloc[-2]
        mom_pct        = (latest_month['total'] - prev_month['total']) / prev_month['total'] * 100
        top3_stations  = cluster_df.nlargest(3, 'avg_daily_total')[['staName','avg_daily_total']].values.tolist()
        anom_count     = len(anom_df)
        top5_growth    = cagr_df[cagr_df['cagr_pct'] > 0].head(5)[['staName','cagr_pct']].values.tolist()
        top5_decline   = cagr_df[cagr_df['cagr_pct'] < 0].tail(5)[['staName','cagr_pct']].values.tolist()

        # 車種近一年佔比
        yr_tt = tt[tt['year'] == tt['year'].max()]
        total_yr = yr_tt['pax_total'].sum()
        local_share  = yr_tt['pax_local'].sum()  / total_yr * 100  if total_yr > 0 else 0
        tze_share    = yr_tt['pax_tzechiang'].sum() / total_yr * 100 if total_yr > 0 else 0

        # 集群分布
        cluster_counts = cluster_df['cluster_name'].value_counts().to_dict() if 'cluster_name' in cluster_df.columns else {}

        # ── 章節一：執行摘要 ──────────────────────────────────
        st.markdown("---")
        st.header("第一章　執行摘要")

        col1, col2, col3 = st.columns(3)
        col1.metric("資料涵蓋期間", "2019/04 – 2025/12")
        col2.metric("累計總運量", f"{total_pax/1e8:.2f} 億人次")
        col3.metric("最新月份月增率",
                    latest_month['ym'].strftime('%Y-%m'),
                    f"{'+' if mom_pct>=0 else ''}{mom_pct:.1f}%")

        st.markdown(f"""
**建議撰寫方向：**
- 本報告彙整台灣鐵路局 2019 年至 2025 年間全台 **243 座車站**的每日進出站資料，
  累計總運量達 **{total_pax/1e8:.2f} 億人次**。
- 最新月份（{latest_month['ym'].strftime('%Y年%m月')}）較上月運量
  {'增加' if mom_pct >= 0 else '下降'} **{abs(mom_pct):.1f}%**，
  {'顯示整體需求持續回溫' if mom_pct >= 0 else '需關注短期需求波動原因'}。
- 報告涵蓋車站集群分析、時間趨勢、異常偵測、客運預測及車種結構等面向，
  並於各章提供策略建議。
        """)

        # ── 章節二：車站分群分析 ──────────────────────────────
        st.markdown("---")
        st.header("第二章　車站功能定位與 DNA 分群")
        st.markdown(f"""
**核心發現：**
- 全台 243 座車站依日均運量與週末／平日比值，聚類為四大功能型態：
  {'、'.join([f"**{k}**（{v} 站）" for k,v in cluster_counts.items()]) if cluster_counts else '詳見車站 DNA 分群地圖'}。
- 日均運量前三名：{' > '.join([f"**{s[0]}**（{int(s[1]):,} 人/日）" for s in top3_stations])}。
- 「週末觀光熱點」型車站的週末運量為平日的 1.5 倍以上，呈現明顯季節性脈衝，
  與「都會核心大站」的穩定通勤需求形成鮮明對比。

**建議撰寫方向：**
1. 以分群地圖呈現地理分布，說明各集群在路網中的定位角色。
2. 比較各集群平均日運量、週末比值，歸納各群的服務特性。
3. 建議依集群制訂差異化行銷策略：都會核心站強化月票/通勤方案，
   觀光熱點站發展套票與接駁合作。
        """)

        # ── 章節三：長期趨勢與 COVID-19 衝擊 ─────────────────
        st.markdown("---")
        st.header("第三章　長期運量趨勢與 COVID-19 衝擊分析")
        st.markdown(f"""
**核心發現：**
- 2020 年 3 月起受 COVID-19 疫情影響，全台運量出現顯著衰退，
  2021 年本土疫情高峰期間降至最低谷。
- 2022 年下半年起隨防疫解封逐步回升，2023–2024 年回復至疫前水準。
- 區間列車佔總運量比例約 **{local_share:.1f}%**，為最主要的通勤服務類型；
  自強號佔比約 **{tze_share:.1f}%**，以中長程旅客為主。

**建議撰寫方向：**
1. 繪製 2019–2025 月度趨勢折線圖，標示 COVID-19 影響期間。
2. 計算各年度總運量及恢復率（以 2019 年為基準）。
3. 分析疫情對各車種的差異性衝擊（高鐵替代效應 vs. 通勤韌性）。
4. 建議建立「緊急事件運量預警機制」，以便提前調度資源。
        """)

        # ── 章節四：異常偵測 ──────────────────────────────────
        st.markdown("---")
        st.header("第四章　異常突波偵測與風險預警")

        if anom_count > 0 and len(anom_df.columns) > 0:
            top_anom = anom_df.nlargest(3, 'z_score') if 'z_score' in anom_df.columns else anom_df.head(3)
            anom_preview = top_anom[['staName','trnOpDate','z_score']].rename(
                columns={'staName':'車站','trnOpDate':'日期','z_score':'Z-Score'}
            ) if all(c in top_anom.columns for c in ['staName','trnOpDate','z_score']) else top_anom.head(3)
            st.dataframe(anom_preview, use_container_width=True, hide_index=True)

        st.markdown(f"""
**核心發現：**
- 近 90 天共偵測到 **{anom_count} 筆**異常突波事件（Z-Score > 3）。
- 異常高峰多集中於連續假期（春節、清明、端午、中秋）前後，
  以及重大活動（演唱會、運動賽事）舉辦日。
- Z-Score 超過 5 的極端事件，通常對應單日運量超出預期均值 150% 以上。

**建議撰寫方向：**
1. 列舉近期 Top 5 異常事件，說明可能原因（假期、活動、天災等）。
2. 分析異常的地理分布：觀光站 vs. 通勤站的觸發頻率差異。
3. 建議建立即時預警通知流程，於 Z-Score > 3 時自動發送調度通知。
4. 評估是否需要動態加班車機制因應突發性需求。
        """)

        # ── 章節五：客運量預測 ────────────────────────────────
        st.markdown("---")
        st.header("第五章　短期客運量預測（Prophet 模型）")
        st.markdown(f"""
**核心發現：**
- 採用 Facebook Prophet 模型，針對全台 243 座車站各自訓練時間序列預測模型，
  預測期間為 90 天，並納入台灣國定假日效應。
- 模型同時捕捉「年週期性」（暑假旺季、農曆新年峰值）與「週週期性」（週末出遊特性）。
- 預測信賴區間（80%）可作為容量規劃的上下界參考。

**建議撰寫方向：**
1. 以臺北、桃園、花蓮等代表性車站為例，展示預測曲線與歷史數據的吻合度。
2. 說明模型評估指標（MAE、MAPE）及其業務意涵。
3. 基於 90 天預測，提出班次調配建議（如特定旺季增開班次）。
4. 建議每月重新訓練模型，以反映最新需求趨勢。
        """)

        # ── 章節六：成長趨勢與 CAGR ──────────────────────────
        st.markdown("---")
        st.header("第六章　各站成長趨勢與 CAGR 排行")

        col_g, col_d = st.columns(2)
        with col_g:
            st.markdown("**成長前 5 名（2019→2024）**")
            if top5_growth:
                for name, cagr in top5_growth:
                    st.markdown(f"- **{name}**：年均成長 +{cagr:.1f}%")
            else:
                st.info("無正成長資料")
        with col_d:
            st.markdown("**衰退前 5 名（2019→2024）**")
            if top5_decline:
                for name, cagr in reversed(top5_decline):
                    st.markdown(f"- **{name}**：年均衰退 {cagr:.1f}%")
            else:
                st.info("無衰退資料")

        st.markdown(f"""
**建議撰寫方向：**
1. 分析高成長站點的共同特徵（城市發展、接駁建設、觀光資源等）。
2. 探討衰退站點的原因（人口外移、競爭交通工具、班次調整等）。
3. 建議針對衰退超過 -5% CAGR 的站點啟動「站點活化計畫」，
   包含與地方政府、旅遊業者合作開發在地遊程。
4. 高成長站點應提前規劃擴容方案，避免尖峰時段擁擠影響服務品質。
        """)

        # ── 章節七：車種結構分析 ──────────────────────────────
        st.markdown("---")
        st.header("第七章　車種結構與服務優化")
        st.markdown(f"""
**核心發現：**
- 區間列車為通勤骨幹，佔總運量約 **{local_share:.1f}%**；
  疫情期間其韌性明顯高於長途車種，顯示在地通勤需求相對穩定。
- 自強號（含太魯閣/普悠瑪）佔比約 **{tze_share:.1f}%**，
  平均旅程距離最長，是台鐵長途競爭力的關鍵。
- 莒光號客運量自 2020 年起持續下滑，市場份額受自強號及高鐵雙面擠壓。

**建議撰寫方向：**
1. 繪製 2019–2025 年各車種市佔率演變堆疊面積圖。
2. 分析莒光號衰退趨勢，評估是否需調整票價策略或服務定位。
3. 比較各車種平均旅程距離的年度變化，推測旅客行為轉變。
4. 建議研究「區間快車」等新車種引入的可行性，填補通勤與長途之間的市場缺口。
        """)

        # ── 章節八：綜合策略建議 ──────────────────────────────
        st.markdown("---")
        st.header("第八章　綜合策略建議")
        st.markdown("""
根據以上各章分析，提出以下六項核心策略建議：

| # | 建議方向 | 優先程度 | 對應分析模組 |
|---|---------|---------|------------|
| 1 | **差異化行銷策略**：依車站 DNA 集群制訂不同促銷方案（通勤月票、觀光套票） | 🔴 高 | 車站 DNA 分群 |
| 2 | **動態班次調配**：建立以預測為基礎的班次調整機制，提升尖峰效率 | 🔴 高 | 運量預測 |
| 3 | **衰退站點活化**：針對 CAGR < -5% 站點啟動跨域合作，開發新客群 | 🟡 中 | CAGR 排行 |
| 4 | **莒光號轉型評估**：研究重新定位或票價調整以提升競爭力 | 🟡 中 | 車種別分析 |
| 5 | **即時預警系統**：建立 Z-Score 自動警示流程，縮短突發事件應變時間 | 🟡 中 | 異常偵測 |
| 6 | **數據治理強化**：提升資料更新頻率至 T+1，並建立資料品質監控機制 | 🟢 低 | 全系統 |

---

> **免責聲明**：本頁建議由系統自動依資料生成，僅供參考。實際策略決策應結合業務專家判斷、法規環境及組織資源進行綜合評估。
        """)

        # ── 報告大綱快速複製區 ────────────────────────────────
        st.markdown("---")
        st.header("📄 報告大綱（可直接複製）")
        outline_text = f"""台鐵營運分析報告大綱
資料期間：2019/04 – 2025/12　　累計運量：{total_pax/1e8:.2f} 億人次

第一章　執行摘要
  1.1 報告背景與目的
  1.2 資料來源與範圍說明
  1.3 主要發現一覽

第二章　車站功能定位與 DNA 分群
  2.1 分群方法（K-Means 聚類）
  2.2 四大車站類型特徵比較
  2.3 地理分布與路網意涵

第三章　長期運量趨勢與 COVID-19 衝擊分析
  3.1 2019–2025 年度總運量演變
  3.2 COVID-19 衝擊期間（2020–2021）量化分析
  3.3 復甦路徑與現況評估

第四章　異常突波偵測與風險預警
  4.1 Z-Score 異常偵測方法說明
  4.2 近 90 天異常事件彙整（共 {anom_count} 筆）
  4.3 異常事件成因分類

第五章　短期客運量預測（Prophet 模型）
  5.1 模型架構與假日效應設計
  5.2 代表性車站預測結果展示
  5.3 模型準確度評估
  5.4 未來 90 天運量展望

第六章　各站成長趨勢與 CAGR 排行
  6.1 CAGR 計算方法
  6.2 高成長站點特徵分析
  6.3 衰退站點成因探討
  6.4 活化建議

第七章　車種結構與服務優化
  7.1 各車種市佔率演變
  7.2 平均旅程距離分析
  7.3 莒光號競爭力評估
  7.4 新車種引入可行性

第八章　綜合策略建議
  8.1 差異化行銷策略
  8.2 動態班次調配機制
  8.3 衰退站點活化計畫
  8.4 即時預警系統建置
  8.5 資料治理強化路徑

附錄
  A. 資料欄位說明
  B. 統計方法補充說明
  C. 各站 CAGR 完整排行表
"""
        st.text_area("報告大綱（點選後 Ctrl+A 全選複製）", outline_text, height=420)
