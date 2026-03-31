import pandas as pd
from sqlalchemy import create_engine, text
import time
import os
import glob

# ==========================================
# 1. 建立資料庫連線
# ==========================================
db_url = 'mysql+pymysql://root:@localhost:3306/TRA_DataMining'
engine = create_engine(db_url)

# ⭐️ 建立萬能翻譯字典：對付政府不同年份亂改的欄位名稱
# 鍵(Key)為全大寫，值(Value)對應到我們資料庫的標準欄位
COLUMN_MAPPING = {
    'TRNOPDATE': 'trnOpDate',
    'TRN_OP_DATE': 'trnOpDate',
    'STACODE': 'staCode',
    'STA_CODE': 'staCode',
    'GATELNCOMINGCUT': 'gateInComingCnt',  # 早年拼字錯誤的版本
    'GATEINCOMINGCNT': 'gateInComingCnt',
    'GATE_IN_COMING_CNT': 'gateInComingCnt',
    'GATEOUTGOINGCNT': 'gateOutGoingCnt',
    'GATE_OUT_GOING_CNT': 'gateOutGoingCnt'
}

# ==========================================
# 階段一：自動從 CSV 補齊全台車站代碼 (動態辨識版)
# ==========================================
def auto_fill_stations(csv_files):
    print("正在掃描 CSV 檔案，自動補齊全台車站名單...")
    all_stations = set()
    
    for file in csv_files:
        if "api" not in os.path.basename(file).lower():
            try:
                # 讀取標題列
                df_header = pd.read_csv(file, nrows=0)
                
                # 尋找哪個欄位是車站代碼 (轉換成大寫後比對)
                sta_col = None
                for col in df_header.columns:
                    if col.strip().upper() in ['STACODE', 'STA_CODE']:
                        sta_col = col
                        break
                        
                if sta_col is None:
                    print(f"❌ 警告：檔案 {os.path.basename(file)} 找不到車站欄位！")
                    continue
                    
                # 只讀取車站代碼欄位來加速處理
                df = pd.read_csv(file, usecols=[sta_col])
                codes = df[sta_col].astype(str).str.zfill(4).unique()
                all_stations.update(codes)
                
            except Exception as e:
                print(f"讀取 {os.path.basename(file)} 時發生錯誤：{e}")
                
    # 寫入資料庫
    with engine.begin() as conn:
        for code in all_stations:
            sql = text(f"INSERT IGNORE INTO StationInfo (staCode, staName) VALUES ('{code}', '車站_{code}')")
            conn.execute(sql)
            
    print(f"✅ 成功補齊 {len(all_stations)} 個車站至基本資料表！\n")

# ==========================================
# 階段二：批次寫入每日進出站人數 (DailyPassenger)
# ==========================================
def insert_daily_passenger(csv_file_path):
    print(f"開始讀取並寫入進出站資料：{os.path.basename(csv_file_path)}")
    start_time = time.time()
    chunk_size = 50000  
    
    total_inserted = 0
    
    for chunk in pd.read_csv(csv_file_path, chunksize=chunk_size):
        # 1. 把讀進來的欄位名稱統一轉成大寫並去除兩側空白
        chunk.rename(columns=lambda x: str(x).strip().upper(), inplace=True)
        
        # 2. 套用萬能翻譯字典，將名稱統一為資料庫標準格式
        chunk.rename(columns=COLUMN_MAPPING, inplace=True)
        
        # 3. 欄位清洗與轉型
        chunk['trnOpDate'] = pd.to_datetime(chunk['trnOpDate'].astype(str)).dt.date
        chunk['staCode'] = chunk['staCode'].astype(str).str.zfill(4)
        
        # 4. 批次寫入 MySQL
        chunk.to_sql(name='DailyPassenger', con=engine, if_exists='append', index=False)
        
        total_inserted += len(chunk)
        print(f"  ...已成功寫入 {total_inserted} 筆資料")

    end_time = time.time()
    print(f"✅ 檔案寫入完成！耗時：{round(end_time - start_time, 2)} 秒\n")

# ==========================================
# 主程式執行區塊
# ==========================================
if __name__ == "__main__":
    folder_path = './每日各站進出站人數20190423-20251231/' 
    all_csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
    all_csv_files.sort()
    
    print(f"找到 {len(all_csv_files)} 個 CSV 檔案準備處理...")
    
    # 1. 執行自動補齊
    auto_fill_stations(all_csv_files)
    
    # 2. 依序匯入資料
    for file_path in all_csv_files:
        if "api" not in os.path.basename(file_path).lower():
            insert_daily_passenger(file_path)
            
    print("🎉 所有年份資料皆已成功寫入資料庫！")