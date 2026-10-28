
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from e_commerce_voice_bot.logger import get_logger

logger = get_logger(__name__)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Load and cache the embedding model."""
    logger.info("Loading embedding model: %s", MODEL_NAME)

    try:
        model = SentenceTransformer(MODEL_NAME)
        logger.info("Embedding model loaded successfully.")
        return model
    except Exception:
        logger.exception("Failed to load embedding model.")
        raise


def generate_embeddings(texts: list[str]):
    """Generate normalized embeddings for product texts."""
    if not texts:
        return []

    model = get_embedding_model()

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    return embeddings
