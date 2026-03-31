"""
將「台鐵各車站詳細資料.json」的經緯度與縣市資訊匯入 StationInfo 表
執行一次即可：python import_station_geodata.py
"""
import sys, json
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from sqlalchemy import create_engine, text

DB_URL = 'mysql+pymysql://root:@localhost:3306/TRA_DataMining'
engine = create_engine(DB_URL)

with open('./台鐵各車站詳細資料.json', encoding='utf-8') as f:
    stations = json.load(f)

print(f"讀取到 {len(stations)} 筆車站資料，開始匯入...")

with engine.begin() as conn:
    # 新增欄位（已存在則略過）
    for ddl in [
        "ALTER TABLE StationInfo ADD COLUMN lat DOUBLE DEFAULT NULL",
        "ALTER TABLE StationInfo ADD COLUMN lon DOUBLE DEFAULT NULL",
        "ALTER TABLE StationInfo ADD COLUMN city VARCHAR(20) DEFAULT NULL",
        "ALTER TABLE StationInfo ADD COLUMN station_class TINYINT DEFAULT NULL",
    ]:
        try:
            conn.execute(text(ddl))
        except Exception:
            pass

    updated = 0
    for s in stations:
        code = str(s.get('StationID', '')).zfill(4)
        pos  = s.get('StationPosition', {})
        lat  = pos.get('PositionLat')
        lon  = pos.get('PositionLon')
        city = s.get('LocationCity', '')
        cls  = s.get('StationClass')

        conn.execute(
            text("""UPDATE StationInfo
                    SET lat=:lat, lon=:lon, city=:city, station_class=:cls
                    WHERE staCode=:code"""),
            {'lat': lat, 'lon': lon, 'city': city, 'cls': cls, 'code': code}
        )
        updated += 1

print(f"✅ 已更新 {updated} 個車站的座標與縣市資訊！")
