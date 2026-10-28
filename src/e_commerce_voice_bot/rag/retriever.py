
import json

import numpy as np

from e_commerce_voice_bot.logger import get_logger
from e_commerce_voice_bot.rag.embeddings import generate_embeddings
from e_commerce_voice_bot.rag.vector_store import load_vector_store

logger = get_logger(__name__)


def retrieve_products(query: str, top_k: int = 5) -> list[dict]:
    """Retrieve semantically similar products using FAISS and SQLite."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")

    query_embedding = generate_embeddings([query])
    query_vector = np.asarray(query_embedding, dtype=np.float32)

    if query_vector.ndim != 2 or query_vector.shape[0] != 1:
        raise ValueError("Query embedding must contain exactly one vector.")

    if not np.isfinite(query_vector).all():
        raise ValueError("Query embedding contains invalid values.")

    index, connection = load_vector_store()

    try:
        if index.ntotal == 0:
            return []

        scores, indices = index.search(
            query_vector,
            min(top_k, index.ntotal),
        )

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
                logger.warning(
                    "No metadata found for vector ID %s.", vector_id
                )
                continue

            try:
                metadata = json.loads(row["metadata_json"])
            except (json.JSONDecodeError, TypeError):
                logger.exception(
                    "Invalid metadata JSON for vector ID %s.", vector_id
                )
                continue

            results.append(
                {
                    "id": row["product_id"],
                    "metadata": metadata,
                    "similarity_score": float(score),
                }
            )

        logger.info("Retrieved %s products for semantic search.", len(results))
        return results

    except Exception:
        logger.exception("Product retrieval failed.")
        raise

    finally:
        connection.close()
