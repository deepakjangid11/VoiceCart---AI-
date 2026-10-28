
import json
import sys

import faiss
import numpy as np

from e_commerce_voice_bot.database import get_db_connection
from e_commerce_voice_bot.exception import CustomException
from e_commerce_voice_bot.logger import get_logger
from e_commerce_voice_bot.rag.chunking import chunk_products
from e_commerce_voice_bot.rag.embeddings import generate_embeddings
from e_commerce_voice_bot.rag.vector_store import (
    INDEX_FILE,
    METADATA_DB,
    N_LIST,
    N_PROBE,
    VECTOR_STORE_DIR,
    create_metadata_db,
)

logger = get_logger(__name__)

BATCH_SIZE = 128
TRAINING_SAMPLE_SIZE = 50_000


def build_product_index() -> None:
    """Build a FAISS IVF index and SQLite metadata in batches."""
    connection = None
    cursor = None
    sample_cursor = None
    metadata_connection = None

    temporary_index = INDEX_FILE.with_suffix(".tmp")
    temporary_metadata = METADATA_DB.with_name(
        f"{METADATA_DB.stem}.tmp.db"
    )

    backup_index = INDEX_FILE.with_suffix(".bak")
    backup_metadata = METADATA_DB.with_name(
        f"{METADATA_DB.stem}.bak.db"
    )

    index_backed_up = False
    metadata_backed_up = False
    new_index_installed = False
    new_metadata_installed = False

    try:
        VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

        # Remove leftovers from an interrupted previous build.
        temporary_index.unlink(missing_ok=True)
        temporary_metadata.unlink(missing_ok=True)
        backup_index.unlink(missing_ok=True)
        backup_metadata.unlink(missing_ok=True)

        connection = get_db_connection()

        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT COUNT(*) AS total FROM products")
        total_products = int(cursor.fetchone()["total"])
        cursor.close()
        cursor = None

        if total_products == 0:
            raise ValueError("No products found in the database.")

        logger.info("Preparing to index %s products.", total_products)

        # Use a deterministic sample instead of ORDER BY RAND(),
        # which can be expensive on a large MySQL table.
        sample_cursor = connection.cursor(dictionary=True)
        sample_cursor.execute(
            """
            SELECT asin, title, category_name, price, stars, reviews
            FROM products
            ORDER BY asin
            LIMIT %s
            """,
            (min(TRAINING_SAMPLE_SIZE, total_products),),
        )
        sample_rows = sample_cursor.fetchall()
        sample_cursor.close()
        sample_cursor = None

        sample_chunks = chunk_products(sample_rows)
        del sample_rows

        if len(sample_chunks) < 2:
            raise ValueError(
                "Not enough valid products to train the FAISS index."
            )

        sample_embeddings = np.asarray(
            generate_embeddings(
                [chunk["text"] for chunk in sample_chunks]
            ),
            dtype=np.float32,
        )

        if (
            sample_embeddings.ndim != 2
            or not np.isfinite(sample_embeddings).all()
        ):
            raise ValueError("Training embeddings are invalid.")

        dimension = sample_embeddings.shape[1]

        # Keep the number of IVF clusters appropriate for the sample size.
        n_list = min(N_LIST, max(1, len(sample_embeddings) // 39))

        quantizer = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIVFFlat(
            quantizer,
            dimension,
            n_list,
            faiss.METRIC_INNER_PRODUCT,
        )

        logger.info(
            "Training FAISS IVF index with %s vectors and %s clusters.",
            len(sample_embeddings),
            n_list,
        )
        index.train(sample_embeddings)
        index.nprobe = min(N_PROBE, n_list)

        del sample_embeddings
        del sample_chunks

        # Build metadata in a temporary database so an existing index
        # remains available if indexing fails.
        metadata_connection = create_metadata_db(temporary_metadata)

        cursor = connection.cursor(dictionary=True)
        cursor.execute(
            """
            SELECT asin, title, category_name, price, stars, reviews
            FROM products
            ORDER BY asin
            """
        )

        processed = 0

        while True:
            rows = cursor.fetchmany(BATCH_SIZE)
            if not rows:
                break

            chunks = chunk_products(rows)
            del rows

            if not chunks:
                continue

            embeddings = np.asarray(
                generate_embeddings([chunk["text"] for chunk in chunks]),
                dtype=np.float32,
            )

            if (
                embeddings.ndim != 2
                or embeddings.shape != (len(chunks), dimension)
                or not np.isfinite(embeddings).all()
            ):
                raise ValueError("Product batch embeddings are invalid.")

            start_id = index.ntotal
            index.add(embeddings)

            metadata_connection.executemany(
                """
                INSERT INTO products (vector_id, product_id, metadata_json)
                VALUES (?, ?, ?)
                """,
                [
                    (
                        start_id + offset,
                        str(chunk["id"]),
                        json.dumps(
                            chunk["metadata"],
                            default=str,
                            ensure_ascii=False,
                        ),
                    )
                    for offset, chunk in enumerate(chunks)
                ],
            )
            metadata_connection.commit()

            processed += len(chunks)

            if processed % (BATCH_SIZE * 10) < len(chunks):
                logger.info(
                    "Indexed %s / %s products.", processed, total_products
                )

            del embeddings
            del chunks

        cursor.close()
        cursor = None

        metadata_connection.close()
        metadata_connection = None

        if index.ntotal != processed:
            raise RuntimeError(
                f"Index count mismatch: {index.ntotal} vectors "
                f"but {processed} products processed."
            )

        # Validate the temporary metadata before replacing existing files.
        check_connection = create_metadata_db(temporary_metadata)
        try:
            metadata_count = check_connection.execute(
                "SELECT COUNT(*) FROM products"
            ).fetchone()[0]
        finally:
            check_connection.close()

        if metadata_count != index.ntotal:
            raise RuntimeError(
                f"Metadata count mismatch: {metadata_count} rows "
                f"for {index.ntotal} vectors."
            )

        faiss.write_index(index, str(temporary_index))

        # Back up the current pair, then install the new pair.
        if INDEX_FILE.exists():
            INDEX_FILE.replace(backup_index)
            index_backed_up = True

        if METADATA_DB.exists():
            METADATA_DB.replace(backup_metadata)
            metadata_backed_up = True

        temporary_index.replace(INDEX_FILE)
        new_index_installed = True

        temporary_metadata.replace(METADATA_DB)
        new_metadata_installed = True

        backup_index.unlink(missing_ok=True)
        backup_metadata.unlink(missing_ok=True)

        logger.info(
            "Product indexing completed successfully: %s products.",
            processed,
        )

    except Exception as e:
        logger.exception("Failed to build product vector index.")

        # Roll back the installed files if finalization fails.
        if new_index_installed:
            INDEX_FILE.unlink(missing_ok=True)
        if new_metadata_installed:
            METADATA_DB.unlink(missing_ok=True)

        if index_backed_up and backup_index.exists():
            backup_index.replace(INDEX_FILE)
        if metadata_backed_up and backup_metadata.exists():
            backup_metadata.replace(METADATA_DB)

        raise CustomException(e, sys) from e

    finally:
        if cursor is not None:
            cursor.close()
        if sample_cursor is not None:
            sample_cursor.close()
        if connection is not None:
            connection.close()
        if metadata_connection is not None:
            metadata_connection.close()

        temporary_index.unlink(missing_ok=True)
        temporary_metadata.unlink(missing_ok=True)


if __name__ == "__main__":
    build_product_index()
