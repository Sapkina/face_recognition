import sqlite3
import pickle
from config import DB_PATH

def load_all_embeddings(db_path=DB_PATH):
    """
    Загружает все эмбеддинги из базы данных.
    Возвращает список: [(name, role, group_name, embedding), ...]
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT name, embedding, role, group_name FROM university")
    records = cursor.fetchall()

    known_faces = []
    for name, embedding_blob, role, group in records:
        embedding = pickle.loads(embedding_blob)
        known_faces.append((name, role, group, embedding))

    cursor.close()
    conn.close()

    return known_faces
