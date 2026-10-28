
import json

import numpy as np

from e_commerce_voice_bot.rag import retriever
from e_commerce_voice_bot.rag.chunking import chunk_products
from e_commerce_voice_bot.rag.embeddings import generate_embeddings
from e_commerce_voice_bot.rag.vector_store import (
    build_vector_store,
    load_vector_store,
)


def test_rag_pipeline(tmp_path, monkeypatch):
    products = [
        {
            "asin": "TEST001",
            "title": "Affordable laptop for college students",
            "category_name": "Computers",
            "price": 450.0,
            "stars": 4.5,
            "reviews": 120,
        },
        {
            "asin": "TEST002",
            "title": "Wireless Bluetooth headphones",
            "category_name": "Electronics",
            "price": 35.0,
            "stars": 4.3,
            "reviews": 80,
        },
        {
            "asin": "TEST003",
            "title": "Gaming laptop with powerful graphics",
            "category_name": "Computers",
            "price": 900.0,
            "stars": 4.7,
            "reviews": 200,
        },
        {
            "asin": "TEST004",
            "title": "Stainless steel water bottle",
            "category_name": "Home",
            "price": 20.0,
            "stars": 4.2,
            "reviews": 45,
        },
        {
            "asin": "TEST005",
            "title": "Laptop stand for desk",
            "category_name": "Computers",
            "price": 28.0,
            "stars": 4.4,
            "reviews": 60,
        },
    ]

    chunks = chunk_products(products)

    assert len(chunks) == 5
    assert "Affordable laptop" in chunks[0]["text"]

    embeddings = np.asarray(
        generate_embeddings([chunk["text"] for chunk in chunks]),
        dtype=np.float32,
    )

    assert embeddings.shape[0] == 5
    assert np.isfinite(embeddings).all()

    from e_commerce_voice_bot.rag import vector_store

    monkeypatch.setattr(vector_store, "VECTOR_STORE_DIR", tmp_path)
    monkeypatch.setattr(vector_store, "INDEX_FILE", tmp_path / "products.index")
    monkeypatch.setattr(
        vector_store,
        "METADATA_DB",
        tmp_path / "products_metadata.db",
    )

    build_vector_store(embeddings, chunks)

    index, connection = load_vector_store()

    try:
        assert index.ntotal == 5

        metadata_count = connection.execute(
            "SELECT COUNT(*) FROM products"
        ).fetchone()[0]

        assert metadata_count == 5

        query_embedding = np.asarray(
            generate_embeddings(["laptop for college"]),
            dtype=np.float32,
        )

        scores, indices = index.search(query_embedding, 3)

        assert len(indices[0]) == 3
        assert all(index_id >= 0 for index_id in indices[0])
        assert np.isfinite(scores).all()

    finally:
        connection.close()


def test_retrieve_products_returns_results(monkeypatch):
    fake_metadata = [
        {
            "vector_id": 0,
            "product_id": "TEST001",
            "metadata_json": json.dumps({"asin": "TEST001"}),
        },
        {
            "vector_id": 1,
            "product_id": "TEST002",
            "metadata_json": json.dumps({"asin": "TEST002"}),
        },
    ]

    class FakeConnection:
        def execute(self, query, params):
            vector_id = params[0]
            row = fake_metadata[vector_id]
            self._product_id = row["product_id"]
            self._metadata_json = row["metadata_json"]
            return self

        def fetchone(self):
            return {
                "product_id": self._product_id,
                "metadata_json": self._metadata_json,
            }

        def close(self):
            pass

    class FakeIndex:
        ntotal = 2

        def search(self, query_vector, top_k):
            return (
                np.array([[0.95, 0.20]], dtype=np.float32),
                np.array([[0, 1]], dtype=np.int64),
            )

    monkeypatch.setattr(
        retriever,
        "load_vector_store",
        lambda: (FakeIndex(), FakeConnection()),
    )
    monkeypatch.setattr(
        retriever,
        "generate_embeddings",
        lambda texts: [[0.1, 0.2]],
    )

    results = retriever.retrieve_products("laptop", top_k=2)

    assert len(results) == 2
    assert results[0]["id"] == "TEST001"
    assert results[0]["metadata"]["asin"] == "TEST001"
    assert abs(results[0]["similarity_score"] - 0.95) < 1e-5
