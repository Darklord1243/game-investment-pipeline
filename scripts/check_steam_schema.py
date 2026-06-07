import sqlite3

conn = sqlite3.connect("data/game_metrics.db")
cols = [r[1] for r in conn.execute("PRAGMA table_info(steam_metrics)").fetchall()]
indexes = conn.execute(
    "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='steam_metrics'"
).fetchall()
print("columns:", cols)
print("indexes:", indexes)
