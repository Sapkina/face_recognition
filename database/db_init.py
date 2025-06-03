import sqlite3
from config import DB_PATH

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("""
    CREATE TABLE IF NOT EXISTS university (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        photo BLOB NOT NULL,
        role TEXT,
        group_name TEXT,
        embedding BLOB  -- будем хранить сериализованный список через pickle
    );
""")

conn.commit()
conn.close()
print('Success')