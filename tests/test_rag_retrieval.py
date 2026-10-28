import json
import sqlite3
import sys
import time
from pathlib import Path

import faiss
import numpy as np

from e_commerce_voice_bot.exception import CustomException
from e_commerce_voice_bot.logger import get_logger
from e_commerce_voice_bot.rag.embeddings import generate_embeddings

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_VECTOR_DIR = PROJECT_ROOT / "data" / "vector_store_test"
TEST_INDEX_FILE = TEST_VECTOR_DIR / "test_products.index"
TEST_METADATA_DB = TEST_VECTOR_DIR / "test_metadata.db"


def validate_rag_retrieval(
    query: str = "bird guide",
    top_k: int = 3,
) -> list[dict]:
    """Validate RAG retrieval using the small test index and measure latencies."""
    logger.info("Starting RAG retrieval validation | query=%s | top_k=%d", query, top_k)

    if not TEST_INDEX_FILE.is_file():
        error_msg = f"Test FAISS index not found: {TEST_INDEX_FILE}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    if not TEST_METADATA_DB.is_file():
        error_msg = f"Test metadata database not found: {TEST_METADATA_DB}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    connection = None
    try:
        # 1. Load test FAISS index
        logger.info("Loading test FAISS index from: %s", TEST_INDEX_FILE)
        index = faiss.read_index(str(TEST_INDEX_FILE))

        # 2. Connect to test SQLite metadata database
        logger.info("Connecting to test metadata DB: %s", TEST_METADATA_DB)
        connection = sqlite3.connect(TEST_METADATA_DB)
        connection.row_factory = sqlite3.Row

        # Verify index count matches metadata count
        meta_count = connection.execute("SELECT COUNT(*) FROM products").fetchone()[0]
        if index.ntotal != meta_count:
            raise ValueError(
                f"Count mismatch: FAISS index has {index.ntotal} vectors, "
                f"SQLite has {meta_count} rows."
            )

        # 3. Measure embedding generation time
        start_time = time.perf_counter()
        t0_embed = time.perf_counter()
        query_embedding = generate_embeddings([query])
        embedding_time = time.perf_counter() - t0_embed

        query_vector = np.asarray(query_embedding, dtype=np.float32)
        if query_vector.ndim != 2 or query_vector.shape[0] != 1:
            raise ValueError("Query embedding must be 2D with a single vector.")

        # 4. Measure FAISS similarity search time
        t0_search = time.perf_counter()
        search_k = min(top_k, index.ntotal)
        scores, indices = index.search(query_vector, search_k)
        vector_search_time = time.perf_counter() - t0_search

        # 5. Retrieve matching metadata from SQLite
        results = []
        for score, vector_id in zip(scores[0], indices[0]):
            if vector_id < 0:
                continue

            row = connection.execute(
                """
                SELECT product_id, metadata_json
                FROM products
                WHERE vector_id = ?
                """,
                (int(vector_id),),
            ).fetchone()

            if row is None:
                logger.warning("No metadata row for vector ID %s", vector_id)
                continue

            metadata = json.loads(row["metadata_json"])
            results.append(
                {
                    "product_id": row["product_id"],
                    "similarity_score": float(score),
                    "metadata": metadata,
                }
            )

        total_retrieval_time = time.perf_counter() - start_time

        # 6. Validate results
        if not results:
            raise ValueError("Retrieval validation failed: No matching products returned.")

        for rank, item in enumerate(results, start=1):
            meta = item.get("metadata", {})
            asin = meta.get("asin")
            title = meta.get("title")

            if not asin or not str(asin).strip():
                raise ValueError(f"Result rank {rank} is missing required 'asin' field.")

            if not title or not str(title).strip():
                raise ValueError(f"Result rank {rank} is missing required 'title' field.")

        # 7. Log required metrics
        logger.info(
            "RAG retrieval validation metrics | "
            "embedding_time=%.4fs | vector_search_time=%.4fs | total_retrieval_time=%.4fs",
            embedding_time,
            vector_search_time,
            total_retrieval_time,
        )
        logger.info(
            "RAG retrieval validation successful | results_returned=%d | top_match=%s",
            len(results),
            results[0]["metadata"].get("title"),
        )

        # 8. Print clear PASS/FAIL summary
        print("\n================ RAG RETRIEVAL VALIDATION SUMMARY ================")
        print("Status               : PASS")
        print(f"Sample Query         : \"{query}\"")
        print(f"Index File           : {TEST_INDEX_FILE.name} ({index.ntotal} vectors)")
        print(f"Results Returned     : {len(results)}")
        print(f"Embedding Time       : {embedding_time * 1000:.2f} ms")
        print(f"Vector Search Time   : {vector_search_time * 1000:.2f} ms")
        print(f"Total Retrieval Time : {total_retrieval_time * 1000:.2f} ms")
        print("------------------------------------------------------------------")
        print("Retrieved Products:")
        for idx, item in enumerate(results, start=1):
            m = item["metadata"]
            print(
                f"  [{idx}] ASIN: {m.get('asin')} | Score: {item['similarity_score']:.4f} | "
                f"Title: {m.get('title')[:60]}"
            )
        print("==================================================================\n")

        return results

    except Exception as e:
        logger.exception("RAG retrieval validation failed: %s", e)
        print("\n================ RAG RETRIEVAL VALIDATION SUMMARY ================")
        print("Status               : FAIL")
        print(f"Error                : {e}")
        print("==================================================================\n")
        raise CustomException(e, sys) from e

    finally:
        if connection:
            connection.close()


def test_rag_retrieval():
    """Pytest test case for RAG retrieval on the small test index."""
    results = validate_rag_retrieval(query="bird guide", top_k=3)
    assert len(results) > 0
    assert "asin" in results[0]["metadata"]
    assert "title" in results[0]["metadata"]
    assert results[0]["metadata"]["asin"] == "0007268149"


def test_rag_retrieval_second_query():
    """Pytest test case verifying a different query against the test index."""
    results = validate_rag_retrieval(query="building construction", top_k=2)
    assert len(results) > 0
    assert "asin" in results[0]["metadata"]
    assert "title" in results[0]["metadata"]
    assert results[0]["metadata"]["asin"] == "0021402442"


def main() -> None:
    try:
        validate_rag_retrieval()
    except CustomException:
        sys.exit(1)


if __name__ == "__main__":
    main()
