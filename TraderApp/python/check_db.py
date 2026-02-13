import sqlite3, os

db_path = "D:/trade/TraderApp/publish_v4/TraderApp.db"
print(f"DB: {db_path} ({os.path.getsize(db_path)} bytes)")
conn = sqlite3.connect(db_path)
c = conn.cursor()
print("\n=== ALL TABLES ===")
c.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
tables = [r[0] for r in c.fetchall()]
for tbl in tables:
    c2 = conn.cursor()
    c2.execute(f"SELECT COUNT(*) FROM [{tbl}]")
    cnt = c2.fetchone()[0]
    print(f"  {tbl}: {cnt} rows")

for tbl in ["positions", "auto_positions", "etf_positions", "broker_positions"]:
    if tbl not in tables:
        print(f"\n{tbl}: does NOT exist")
        continue
    c.execute(f"SELECT * FROM [{tbl}]")
    rows = c.fetchall()
    if rows:
        print(f"\n=== {tbl} ({len(rows)} rows) ===")
        # Get column names
        c.execute(f"PRAGMA table_info([{tbl}])")
        cols = [r[1] for r in c.fetchall()]
        print(f"  Columns: {cols}")
        for r in rows[:10]:
            print(f"  {r}")
    else:
        print(f"\n{tbl}: exists but EMPTY")

conn.close()
