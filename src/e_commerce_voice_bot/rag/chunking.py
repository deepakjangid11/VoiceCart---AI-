
from typing import Any


def product_to_text(product: dict[str, Any]) -> str:
    """Convert a product record into text suitable for embeddings."""
    fields = [
        ("Product", product.get("title")),
        ("Category", product.get("category_name")),
        ("Price", product.get("price")),
        ("Rating", product.get("stars")),
        ("Reviews", product.get("reviews")),
        ("Description", product.get("description")),
    ]

    parts = [
        f"{label}: {value}"
        for label, value in fields
        if value is not None and str(value).strip()
    ]

    return "\n".join(parts)


def chunk_products(
    products: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert product records into embedding-ready documents."""
    chunks = []

    for product in products:
        text = product_to_text(product)

        if not text.strip():
            continue

        chunks.append(
            {
                "id": str(product.get("asin", "")),
                "text": text,
                "metadata": {
                    "asin": product.get("asin"),
                    "title": product.get("title"),
                    "category_name": product.get("category_name"),
                    "price": product.get("price"),
                    "stars": product.get("stars"),
                },
            }
        )

    return chunks
