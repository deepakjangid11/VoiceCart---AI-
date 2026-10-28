
from e_commerce_voice_bot.product_repository import (
    search_products,
    get_product_by_asin,
    get_products_by_category,
    search_products_advanced,
)


def test_search_products():
    products = search_products("iphone", 5)

    assert isinstance(products, list)

    if products:
        assert "asin" in products[0]
        assert "title" in products[0]


def test_get_product_by_asin():
    products = search_products("iphone", 1)

    if not products:
        return

    asin = products[0]["asin"]
    product = get_product_by_asin(asin)

    assert product is not None
    assert product["asin"] == asin


def test_get_products_by_category():
    products = get_products_by_category("Electronics", 5)

    assert isinstance(products, list)


def test_search_products_advanced():
    products = search_products_advanced(
        search_term="iphone",
        max_price=1000,
        min_rating=4,
        limit=5,
    )

    assert isinstance(products, list)

    for product in products:
        assert product["price"] is None or product["price"] <= 1000
        assert product["stars"] is None or product["stars"] >= 4
