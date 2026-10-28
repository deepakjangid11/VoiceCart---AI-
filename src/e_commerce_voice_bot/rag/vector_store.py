
import json
import sqlite3
from pathlib import Path

import faiss
import numpy as np

from e_commerce_voice_bot.logger import get_logger

logger = get_logger(__name__)

VECTOR_STORE_DIR = Path("data/vector_store")
INDEX_FILE = VECTOR_STORE_DIR / "products.index"
METADATA_DB = VECTOR_STORE_DIR / "products_metadata.db"

N_LIST = 4096
N_PROBE = 16


def create_metadata_db(
    db_path: Path | None = None,
) -> sqlite3.Connection:
    """Create a disk-backed SQLite database for product metadata."""
    if db_path is None:
        db_path = METADATA_DB

    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)

    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS products (
                vector_id INTEGER PRIMARY KEY,
                product_id TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            )
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_products_product_id
            ON products(product_id)
            """
        )

        connection.commit()
        return connection

    except Exception:
        connection.close()
        logger.exception("Failed to create metadata database.")
        raise


def build_vector_store(
    embeddings: np.ndarray,
    chunks: list[dict],
) -> None:
    """Build and save a FAISS index with matching SQLite metadata."""
    if len(embeddings) != len(chunks) or not chunks:
        raise ValueError(
            "Embeddings and chunks must be non-empty "
            "and have equal lengths."
        )

    vectors = np.asarray(embeddings, dtype=np.float32)

    if vectors.ndim != 2 or vectors.shape[1] == 0:
        raise ValueError(
            "Embeddings must be a non-empty 2D array."
        )

    if not np.isfinite(vectors).all():
        raise ValueError(
            "Embeddings contain NaN or infinite values."
        )

    # Inner product gives cosine similarity for normalized embeddings.
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

    temporary_index = INDEX_FILE.with_suffix(".tmp")
    temporary_index.unlink(missing_ok=True)

    connection = None

    try:
        faiss.write_index(index, str(temporary_index))

        connection = create_metadata_db()

        with connection:
            connection.execute("DELETE FROM products")

            connection.executemany(
                """
                INSERT INTO products (
                    vector_id,
                    product_id,
                    metadata_json
                )
                VALUES (?, ?, ?)
                """,
                [
                    (
                        vector_id,
                        str(chunk.get("id", "")),
                        json.dumps(
                            chunk.get("metadata", {}),
                            default=str,
                            ensure_ascii=False,
                        ),
                    )
                    for vector_id, chunk in enumerate(chunks)
                ],
            )

        # Install the new index after metadata has been written.
        temporary_index.replace(INDEX_FILE)

        logger.info(
            "Saved FAISS index with %s products.",
            index.ntotal,
        )

    except Exception:
        temporary_index.unlink(missing_ok=True)
        logger.exception("Failed to build vector store.")
        raise

    finally:
        if connection is not None:
            connection.close()


def load_vector_store():
    """Load the FAISS index and its SQLite metadata connection."""
    if not INDEX_FILE.is_file():
        raise FileNotFoundError(
            f"FAISS index not found: {INDEX_FILE}. "
            "Build the product index first."
        )

    if not METADATA_DB.is_file():
        raise FileNotFoundError(
            f"Metadata database not found: {METADATA_DB}. "
            "Build the product index first."
        )

    index = faiss.read_index(str(INDEX_FILE))

    connection = sqlite3.connect(METADATA_DB)
    connection.row_factory = sqlite3.Row

    try:
        metadata_count = connection.execute(
            "SELECT COUNT(*) FROM products"
        ).fetchone()[0]

        if index.ntotal != metadata_count:
            raise ValueError(
                f"Vector store mismatch: FAISS contains "
                f"{index.ntotal} vectors, but SQLite contains "
                f"{metadata_count} metadata rows."
            )

        if isinstance(index, faiss.IndexIVF):
            index.nprobe = min(N_PROBE, index.nlist)

        logger.info(
            "Loaded vector store with %s vectors.",
            index.ntotal,
        )

        return index, connection

    except Exception:
        connection.close()
        logger.exception("Failed to load vector store.")
        raise
