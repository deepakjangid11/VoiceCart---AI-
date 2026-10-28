
import json

from mcp.server import MCPServer

from e_commerce_voice_bot.logger import get_logger
from e_commerce_voice_bot.product_repository import (
    search_products,
    get_product_by_asin,
    get_products_by_category,
    search_products_advanced,
)

mcp = MCPServer("VoiceCart AI")
logger = get_logger(__name__)


@mcp.tool()
def search_products_tool(search_term: str, limit: int = 10) -> str:
    """Search the product catalog by keyword and return matching products as JSON."""
    if not search_term.strip():
        raise ValueError("Search term cannot be empty.")

    if not 1 <= limit <= 50:
        raise ValueError("Limit must be between 1 and 50.")

    try:
        products = search_products(search_term.strip(), limit=limit)
        return json.dumps(products, default=str, ensure_ascii=False)
    except Exception:
        logger.exception("MCP product search failed.")
        raise


@mcp.tool()
def get_product_by_asin_tool(asin: str) -> str:
    """Retrieve a product by its Amazon Standard Identification Number."""
    if not asin.strip():
        raise ValueError("ASIN cannot be empty.")

    try:
        product = get_product_by_asin(asin.strip())
        return json.dumps(product, default=str, ensure_ascii=False)
    except Exception:
        logger.exception("MCP product lookup failed.")
        raise


@mcp.tool()
def get_products_by_category_tool(
    category_name: str,
    limit: int = 10,
) -> str:
    """Retrieve products from a specific product category."""
    if not category_name.strip():
        raise ValueError("Category name cannot be empty.")

    if not 1 <= limit <= 50:
        raise ValueError("Limit must be between 1 and 50.")

    try:
        products = get_products_by_category(
            category_name.strip(),
            limit=limit,
        )
        return json.dumps(products, default=str, ensure_ascii=False)
    except Exception:
        logger.exception("MCP category search failed.")
        raise


@mcp.tool()
def search_products_advanced_tool(
    search_term: str = "",
    min_price: float | None = None,
    max_price: float | None = None,
    min_rating: float | None = None,
    category_name: str | None = None,
    best_seller_only: bool = False,
    sort_by: str = "relevance",
    limit: int = 10,
) -> str:
    """Search products using price, rating, category, bestseller, and sorting filters."""
    if not 1 <= limit <= 50:
        raise ValueError("Limit must be between 1 and 50.")

    if min_price is not None and min_price < 0:
        raise ValueError("Minimum price cannot be negative.")

    if max_price is not None and max_price < 0:
        raise ValueError("Maximum price cannot be negative.")

    if (
        min_price is not None
        and max_price is not None
        and min_price > max_price
    ):
        raise ValueError("Minimum price cannot exceed maximum price.")

    if min_rating is not None and not 0 <= min_rating <= 5:
        raise ValueError("Minimum rating must be between 0 and 5.")

    if sort_by not in {
        "price_low",
        "price_high",
        "rating",
        "reviews",
        "relevance",
    }:
        raise ValueError("Invalid sort_by value.")

    try:
        products = search_products_advanced(
            search_term=search_term.strip(),
            min_price=min_price,
            max_price=max_price,
            min_rating=min_rating,
            category_name=category_name.strip() if category_name else None,
            best_seller_only=best_seller_only,
            sort_by=sort_by,
            limit=limit,
        )
        return json.dumps(products, default=str, ensure_ascii=False)
    except Exception:
        logger.exception("MCP advanced product search failed.")
        raise


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
