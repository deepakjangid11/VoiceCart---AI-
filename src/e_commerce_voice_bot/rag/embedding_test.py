import csv
import itertools
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from e_commerce_voice_bot.exception import CustomException
from e_commerce_voice_bot.logger import get_logger

logger = get_logger(__name__)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
NUM_SAMPLE_PRODUCTS = 10
EXPECTED_DIMENSION = 384


def build_embedding_text(product: dict[str, Any]) -> str:
    """Build a structured text representation of a product for embedding."""
    title = str(product.get("title") or "").strip()
    category_id = str(product.get("category_id") or "").strip()
    price = str(product.get("price") or "").strip()
    stars = str(product.get("stars") or "").strip()

    return (
        f"Product: {title}\n"
        f"Category: {category_id}\n"
        f"Price: {price}\n"
        f"Rating: {stars}"
    )


def load_sample_products(
    csv_path: Path,
    limit: int = NUM_SAMPLE_PRODUCTS,
) -> list[dict[str, str]]:
    """Read only the first `limit` products from the test dataset."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"Test dataset not found at: {csv_path}")

    products: list[dict[str, str]] = []
    with open(csv_path, mode="r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in itertools.islice(reader, limit):
            products.append(row)

    return products


def validate_embeddings(
    embeddings: np.ndarray,
    expected_count: int = NUM_SAMPLE_PRODUCTS,
    expected_dim: int = EXPECTED_DIMENSION,
) -> None:
    """Validate that the generated embeddings meet dimensional and numeric constraints."""
    if not isinstance(embeddings, np.ndarray):
        raise TypeError(
            f"Expected embeddings to be a numpy ndarray, got {type(embeddings).__name__}."
        )

    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected 2D embedding array, got shape {embeddings.shape}."
        )

    count, dim = embeddings.shape

    if count != expected_count:
        raise ValueError(
            f"Expected {expected_count} embeddings, but got {count}."
        )

    if dim != expected_dim:
        raise ValueError(
            f"Expected embedding dimension of {expected_dim}, but got {dim}."
        )

    if not np.issubdtype(embeddings.dtype, np.floating):
        raise TypeError(
            f"Expected floating point dtype, got {embeddings.dtype}."
        )

    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain NaN or infinite values.")

    if np.all(embeddings == 0):
        raise ValueError("Embeddings contain only zero vectors.")


def run_embedding_test() -> np.ndarray:
    """Execute the isolated RAG embedding test on the first 10 products."""
    logger.info("RAG embedding test started.")

    try:
        project_root = Path(__file__).resolve().parents[3]
        csv_path = project_root / "data" / "processed" / "rag_test_products.csv"

        logger.info("Loading sample products from: %s", csv_path)
        products = load_sample_products(csv_path, limit=NUM_SAMPLE_PRODUCTS)

        if len(products) != NUM_SAMPLE_PRODUCTS:
            raise ValueError(
                f"Expected {NUM_SAMPLE_PRODUCTS} products from test dataset, "
                f"found {len(products)}."
            )

        logger.info(
            "Products loaded successfully | count=%d",
            len(products),
        )

        texts = [build_embedding_text(p) for p in products]

        logger.info("Loading embedding model: %s", MODEL_NAME)
        model = SentenceTransformer(MODEL_NAME)
        logger.info("Embedding model loaded successfully.")

        logger.info("Generating embeddings for %d products...", len(texts))
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        logger.info("Embeddings generated successfully.")

        logger.info("Validating generated embeddings...")
        validate_embeddings(
            embeddings,
            expected_count=len(texts),
            expected_dim=EXPECTED_DIMENSION,
        )
        logger.info(
            "Embedding validation successful | count=%d | shape=%s | dtype=%s",
            len(texts),
            embeddings.shape,
            embeddings.dtype,
        )

        print("\n================ RAG EMBEDDING TEST RESULT ================")
        print(f"Products Processed   : {len(products)}")
        print(f"Embedding Shape      : {embeddings.shape}")
        print(f"Embedding Dtype      : {embeddings.dtype}")
        print("Validation Status    : PASSED (384-dim, normalized, finite)")
        print("===========================================================\n")

        return embeddings

    except Exception as e:
        logger.exception("RAG embedding test failed: %s", e)
        raise CustomException(e, sys) from e


def main() -> None:
    try:
        run_embedding_test()
    except CustomException as e:
        print(f"FATAL: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
