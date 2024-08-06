import sqlite3
import sqlite_vec
import struct
from typing import List, Dict, Union, Tuple

def serialize_f32(vector: List[float]) -> bytes:
    """serializes a list of floats into a compact "raw bytes" format"""
    return struct.pack("%sf" % len(vector), *vector)

def deserialize_f32(blob: bytes) -> List[float]:
    """deserializes a blob of bytes into a list of floats"""
    return list(struct.unpack("%sf" % (len(blob) // 4), blob))

class Database:
    def __init__(self, filename: str):
        self.conn = sqlite3.connect(filename)
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)
        self.create_tables()

    def create_tables(self):
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            embedding BLOB NOT NULL
        )
        """)
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS embedding_model (
            id INTEGER PRIMARY KEY,
            model TEXT NOT NULL,
            size INTEGER NOT NULL
        )
        """)
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_embeddings ON embeddings(embedding)")
        self.conn.commit()

    def get_embedding_model(self) -> Tuple[str, int]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT model, size FROM embedding_model LIMIT 1")
        result = cursor.fetchone()
        return result if result else (None, None)

    def set_embedding_model(self, model: str, size: int):
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM embedding_model")
        cursor.execute("INSERT INTO embedding_model (model, size) VALUES (?, ?)", (model, size))
        self.conn.commit()

    def insert_embeddings(self, texts: List[str], embeddings: List[Tuple[List[float], str, int]], ids: List[Union[int, None]]) -> List[int]:
        cursor = self.conn.cursor()
        result_ids = []
        current_model, current_size = self.get_embedding_model()
        
        for text, (embedding, model, size), id in zip(texts, embeddings, ids):
            if current_model is None and current_size is None:
                self.set_embedding_model(model, size)
                current_model, current_size = model, size
            elif model != current_model or size != current_size:
                raise ValueError(f"Embedding model mismatch. Expected {current_model} with size {current_size}, but got {model} with size {size}")
            
            if id is None:
                cursor.execute("INSERT INTO embeddings (text, embedding) VALUES (?, ?)", (text, serialize_f32(embedding)))
                id = cursor.lastrowid
            else:
                cursor.execute("INSERT OR REPLACE INTO embeddings (id, text, embedding) VALUES (?, ?, ?)", (id, text, serialize_f32(embedding)))
            result_ids.append(id)
        self.conn.commit()
        return result_ids

    def search_similar(self, query_embedding: List[float], n: int) -> List[Dict[str, Union[str, float]]]:
        cursor = self.conn.cursor()
        query_blob = serialize_f32(query_embedding)
        cursor.execute("""
        SELECT id, text, embedding
        FROM embeddings
        """)
        results = []
        for row in cursor.fetchall():
            embedding = deserialize_f32(row[2])
            similarity = self.cosine_similarity(query_embedding, embedding)
            results.append({"id": row[0], "text": row[1], "similarity": similarity})
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:n]

    def cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        dot_product = sum(a * b for a, b in zip(v1, v2))
        magnitude1 = sum(a * a for a in v1) ** 0.5
        magnitude2 = sum(b * b for b in v2) ** 0.5
        return dot_product / (magnitude1 * magnitude2)

    def delete_embeddings(self, ids: List[int]):
        placeholders = ','.join('?' * len(ids))
        self.conn.execute(f"DELETE FROM embeddings WHERE id IN ({placeholders})", ids)
        self.conn.commit()
