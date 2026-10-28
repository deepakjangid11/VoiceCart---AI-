import json
import sqlite3
from pathlib import Path

import faiss
import numpy as np

from e_commerce_voice_bot.database import get_db_connection
from e_commerce_voice_bot.rag.chunking import chunk_products
from e_commerce_voice_bot.rag.embeddings import generate_embeddings

TEST_DIR = Path("data/vector_store_test")
INDEX_FILE = TEST_DIR / "test_products.index"
METADATA_DB = TEST_DIR / "test_metadata.db"


def main():
    connection = get_db_connection()

    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute(
            """
            SELECT asin, title, category_name, price, stars, reviews
            FROM products
            ORDER BY asin
            LIMIT 5
            """
        )
        products = cursor.fetchall()
        cursor.close()
    finally:
        connection.close()

    chunks = chunk_products(products)

    if len(chunks) < 2:
        raise ValueError("At least two valid products are required.")

    embeddings = np.asarray(
        generate_embeddings([chunk["text"] for chunk in chunks]),
        dtype=np.float32,
    )

    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain invalid values.")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    TEST_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_FILE))

    with sqlite3.connect(METADATA_DB) as db:
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS products (
                vector_id INTEGER PRIMARY KEY,
                product_id TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            )
            """
        )
        db.execute("DELETE FROM products")
        db.executemany(
            """
            INSERT INTO products
            (vector_id, product_id, metadata_json)
            VALUES (?, ?, ?)
            """,
            [
                (
                    i,
                    chunk["id"],
                    json.dumps(chunk["metadata"], default=str),
                )
                for i, chunk in enumerate(chunks)
            ],
        )

    loaded_index = faiss.read_index(str(INDEX_FILE))

    with sqlite3.connect(METADATA_DB) as db:
        count = db.execute(
            "SELECT COUNT(*) FROM products"
        ).fetchone()[0]

    assert loaded_index.ntotal == len(chunks)
    assert count == len(chunks)

    print(f"SUCCESS: indexed {loaded_index.ntotal} products")
    print(f"FAISS dimension: {loaded_index.d}")
    print(f"Metadata rows: {count}")
    print(f"Test files: {TEST_DIR.resolve()}")


if __name__ == "__main__":
    main()