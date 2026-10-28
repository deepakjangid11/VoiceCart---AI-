import sys

from e_commerce_voice_bot.database import get_db_connection
from e_commerce_voice_bot.exception import CustomException
from e_commerce_voice_bot.logger import get_logger


logger = get_logger(__name__)


def search_products(search_term: str, limit: int = 10):
    connection = None

    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)

        query = """
        SELECT
            asin,
            title,
            price,
            list_price,
            stars,
            reviews,
            category_name,
            is_best_seller,
            bought_in_last_month,
            img_url,
            product_url
        FROM products
        WHERE title LIKE %s
        ORDER BY
            is_best_seller DESC,
            stars DESC,
            reviews DESC
        LIMIT %s
        """

        search_pattern = f"%{search_term}%"

        cursor.execute(
            query,
            (search_pattern, limit)
        )

        products = cursor.fetchall()

        logger.info(
            "Product search completed | term=%s | results=%d",
            search_term,
            len(products)
        )

        return products

    except Exception as e:
        logger.exception(
            "Product search failed | term=%s",
            search_term
        )
        raise CustomException(e, sys)

    finally:
        if connection:
            connection.close()


def get_product_by_asin(asin: str):
    connection = None

    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)

        query = """
        SELECT
            asin,
            title,
            price,
            list_price,
            stars,
            reviews,
            category_name,
            is_best_seller,
            bought_in_last_month,
            img_url,
            product_url
        FROM products
        WHERE asin = %s
        LIMIT 1
        """

        cursor.execute(query, (asin,))

        product = cursor.fetchone()

        logger.info(
            "Product lookup completed | asin=%s | found=%s",
            asin,
            product is not None
        )

        return product

    except Exception as e:
        logger.exception(
            "Product lookup failed | asin=%s",
            asin
        )
        raise CustomException(e, sys)

    finally:
        if connection:
            connection.close()


def get_products_by_category(
    category_name: str,
    limit: int = 10
):
    connection = None

    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)

        query = """
        SELECT
            asin,
            title,
            price,
            list_price,
            stars,
            reviews,
            category_name,
            is_best_seller,
            bought_in_last_month,
            img_url,
            product_url
        FROM products
        WHERE category_name LIKE %s
        ORDER BY
            stars DESC,
            reviews DESC
        LIMIT %s
        """

        category_pattern = f"%{category_name}%"

        cursor.execute(
            query,
            (category_pattern, limit)
        )

        products = cursor.fetchall()

        logger.info(
            "Category search completed | category=%s | results=%d",
            category_name,
            len(products)
        )

        return products

    except Exception as e:
        logger.exception(
            "Category search failed | category=%s",
            category_name
        )
        raise CustomException(e, sys)

    finally:
        if connection:
            connection.close()

def search_products_advanced(
    search_term=None,
    min_price=None,
    max_price=None,
    min_rating=None,
    category_name=None,
    best_seller_only=False,
    sort_by="relevance",
    limit=10
):
    connection = None

    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)

        conditions = []
        parameters = []

        if search_term:
            conditions.append("title LIKE %s")
            parameters.append(f"%{search_term}%")

        if min_price is not None:
            conditions.append("price >= %s")
            parameters.append(min_price)

        if max_price is not None:
            conditions.append("price <= %s")
            parameters.append(max_price)

        if min_rating is not None:
            conditions.append("stars >= %s")
            parameters.append(min_rating)

        if category_name:
            conditions.append("category_name LIKE %s")
            parameters.append(f"%{category_name}%")

        if best_seller_only:
            conditions.append("is_best_seller = TRUE")

        where_clause = ""

        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        sort_options = {
            "price_low": "price ASC",
            "price_high": "price DESC",
            "rating": "stars DESC, reviews DESC",
            "reviews": "reviews DESC",
            "relevance": "is_best_seller DESC, stars DESC, reviews DESC"
        }

        order_clause = sort_options.get(
            sort_by,
            sort_options["relevance"]
        )

        query = f"""
        SELECT
            asin,
            title,
            price,
            list_price,
            stars,
            reviews,
            category_name,
            is_best_seller,
            bought_in_last_month,
            img_url,
            product_url
        FROM products
        {where_clause}
        ORDER BY {order_clause}
        LIMIT %s
        """

        parameters.append(limit)

        cursor.execute(query, tuple(parameters))

        products = cursor.fetchall()

        logger.info(
            "Advanced product search completed | "
            "term=%s | min_price=%s | max_price=%s | "
            "min_rating=%s | category=%s | results=%d",
            search_term,
            min_price,
            max_price,
            min_rating,
            category_name,
            len(products)
        )

        return products

    except Exception as e:
        logger.exception("Advanced product search failed")
        raise CustomException(e, sys)

    finally:
        if connection:
            connection.close()
