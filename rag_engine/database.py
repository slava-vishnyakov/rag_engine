import sqlite3
import sqlite_vec
from typing import List, Dict, Union

class Database:
    def __init__(self, filename: str):
        self.conn = sqlite3.connect(filename)
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)
        self.create_table()

    def create_table(self):
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            embedding BLOB NOT NULL
        )
        """)
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_embeddings ON embeddings(embedding)")
        self.conn.commit()

    def insert_embeddings(self, texts: List[str], embeddings: List[List[float]], ids: List[Union[int, None]]) -> List[int]:
        cursor = self.conn.cursor()
        result_ids = []
        for text, embedding, id in zip(texts, embeddings, ids):
            if id is None:
                cursor.execute("INSERT INTO embeddings (text, embedding) VALUES (?, ?)", (text, sqlite_vec.array(embedding)))
                id = cursor.lastrowid
            else:
                cursor.execute("INSERT OR REPLACE INTO embeddings (id, text, embedding) VALUES (?, ?, ?)", (id, text, sqlite_vec.array(embedding)))
            result_ids.append(id)
        self.conn.commit()
        return result_ids

    def search_similar(self, query_embedding: List[float], n: int) -> List[Dict[str, Union[str, float]]]:
        cursor = self.conn.cursor()
        cursor.execute("""
        SELECT id, text, vec_cosine_similarity(embedding, ?) AS similarity
        FROM embeddings
        ORDER BY similarity DESC
        LIMIT ?
        """, (sqlite_vec.array(query_embedding), n))
        results = cursor.fetchall()
        return [{"id": row[0], "text": row[1], "similarity": row[2]} for row in results]

    def delete_embeddings(self, ids: List[int]):
        placeholders = ','.join('?' * len(ids))
        self.conn.execute(f"DELETE FROM embeddings WHERE id IN ({placeholders})", ids)
        self.conn.commit()
