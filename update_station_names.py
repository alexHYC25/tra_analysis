import json
from sqlalchemy import create_engine, text

# 1. 建立資料庫連線 (請確保密碼與之前設定相同)
db_url = 'mysql+pymysql://root:@localhost:3306/TRA_DataMining'
engine = create_engine(db_url)

def update_real_station_names(json_path):
    print(f"開始解析真實車站 JSON 檔案：{json_path}")
    
    # 讀取 JSON 檔案
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # 解析巢狀結構，整理出不重複的車站字典 
    # 格式會變成：{ '3350': '成功', '0920': '八堵', ... }
    station_dict = {}
    for line in data:
        # 有些路線可能沒有 Stations 欄位，用 .get() 比較安全
        for st in line.get('Stations', []):
            code = str(st['StationID']).zfill(4)
            name = st['StationName']
            station_dict[code] = name
            
    print(f"從 JSON中共解析出 {len(station_dict)} 個不重複的車站。")
    print("準備同步至 MySQL 資料庫...")

    # 寫入資料庫並執行更新
    updated_count = 0
    with engine.begin() as conn:
        for code, name in station_dict.items():
            # 使用 ON DUPLICATE KEY UPDATE，無痛覆蓋掉之前的假名稱
            sql = text(f"""
                INSERT INTO StationInfo (staCode, staName) 
                VALUES ('{code}', '{name}')
                ON DUPLICATE KEY UPDATE staName = '{name}';
            """)
            conn.execute(sql)
            updated_count += 1
            
    print(f"✅ 成功校正並更新了 {updated_count} 個車站的真實中文名稱！")

if __name__ == "__main__":
    # 🔴 請將此處替換成你實際的 JSON 檔案路徑
    json_file_path = './車站基本資料.json' 
    update_real_station_names(json_file_path)