"""
Phase 3：運量預測批次腳本
對全台所有車站訓練 Prophet 模型，預測未來 90 天運量，結果存入 StationForecast 表

執行：python forecasting.py
（可按 Ctrl+C 中斷，已預測的車站不會重複執行）
"""
import sys, warnings
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

import pandas as pd
from prophet import Prophet
from sqlalchemy import create_engine, text

DB_URL = 'mysql+pymysql://root:@localhost:3306/TRA_DataMining'
engine = create_engine(DB_URL)

# ──────────────────────────────────────────────────────────
# 1. 建立 StationForecast 表（首次執行自動建立）
# ──────────────────────────────────────────────────────────
CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS StationForecast (
    staCode   VARCHAR(10)  NOT NULL,
    ds        DATE         NOT NULL,
    yhat      FLOAT,
    yhat_lower FLOAT,
    yhat_upper FLOAT,
    is_future  TINYINT(1)  DEFAULT 0,
    PRIMARY KEY (staCode, ds)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""
with engine.begin() as conn:
    conn.execute(text(CREATE_TABLE))
print("✅ StationForecast 表已就緒\n")

# ──────────────────────────────────────────────────────────
# 2. 取得台灣節假日（傳入 Prophet 作為額外回歸特徵）
# ──────────────────────────────────────────────────────────
def get_taiwan_holidays():
    from workalendar.asia import Taiwan
    cal = Taiwan()
    rows = []
    for year in range(2019, 2027):
        for date, name in cal.holidays(year):
            rows.append({'ds': pd.Timestamp(date), 'holiday': name})
    return pd.DataFrame(rows)

holidays_df = get_taiwan_holidays()
print(f"節假日資料：{len(holidays_df)} 筆（2019–2026）\n")

# ──────────────────────────────────────────────────────────
# 3. 取得待預測車站清單（排除已預測的）
# ──────────────────────────────────────────────────────────
all_stations = pd.read_sql(
    "SELECT staCode, staName FROM StationInfo WHERE cluster_id IS NOT NULL ORDER BY staCode",
    con=engine
)
done_stations = pd.read_sql(
    "SELECT DISTINCT staCode FROM StationForecast WHERE is_future = 1",
    con=engine
)
done_set = set(done_stations['staCode'].tolist())
pending = all_stations[~all_stations['staCode'].isin(done_set)]
print(f"總車站：{len(all_stations)}，已完成：{len(done_set)}，待處理：{len(pending)}\n")

# ──────────────────────────────────────────────────────────
# 4. 逐站訓練 Prophet 並存入資料庫
# ──────────────────────────────────────────────────────────
FORECAST_DAYS = 90

for idx, (_, row) in enumerate(pending.iterrows(), 1):
    code = row['staCode']
    name = row['staName']
    print(f"[{idx}/{len(pending)}] 訓練 {name}（{code}）...", end=' ', flush=True)

    # 讀取歷史資料
    ts = pd.read_sql(
        text("SELECT trnOpDate AS ds, (gateInComingCnt+gateOutGoingCnt) AS y "
             "FROM DailyPassenger WHERE staCode=:code ORDER BY ds"),
        con=engine, params={'code': code}
    )
    ts['ds'] = pd.to_datetime(ts['ds'])

    if len(ts) < 60:
        print("資料不足，跳過")
        continue

    # 訓練 Prophet
    model = Prophet(
        holidays=holidays_df,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10,
        holidays_prior_scale=10,
        interval_width=0.9,
    )
    model.fit(ts)

    # 預測（歷史 + 未來）
    future = model.make_future_dataframe(periods=FORECAST_DAYS)
    forecast = model.predict(future)
    forecast['staCode']   = code
    forecast['is_future'] = (forecast['ds'] > ts['ds'].max()).astype(int)

    save = forecast[['staCode','ds','yhat','yhat_lower','yhat_upper','is_future']].copy()
    save['ds'] = save['ds'].dt.date
    save['yhat']       = save['yhat'].clip(lower=0).round(0)
    save['yhat_lower'] = save['yhat_lower'].clip(lower=0).round(0)
    save['yhat_upper'] = save['yhat_upper'].clip(lower=0).round(0)

    # 寫入資料庫（覆蓋同 staCode 的舊資料）
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM StationForecast WHERE staCode=:code"), {'code': code})
        save.to_sql('StationForecast', con=conn, if_exists='append', index=False)

    print(f"完成（{len(save)} 筆）")

print("\n🎉 全部預測完成！")
