import csv
import sys
from pathlib import Path

from e_commerce_voice_bot.database import get_db_connection
from e_commerce_voice_bot.exception import CustomException
from e_commerce_voice_bot.logger import get_logger

logger = get_logger(__name__)

PROCESSED_DATA_DIR = Path("data/processed")
PRODUCTS_FILE = PROCESSED_DATA_DIR / "products_cleaned.csv"

BATCH_SIZE = 5000


INSERT_QUERY = """
INSERT INTO products (
    asin,
    title,
    img_url,
    product_url,
    stars,
    reviews,
    price,
    list_price,
    category_id,
    category_name,
    is_best_seller,
    bought_in_last_month
)
VALUES (
    %s, %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s, %s
)
ON DUPLICATE KEY UPDATE
    title = VALUES(title),
    img_url = VALUES(img_url),
    product_url = VALUES(product_url),
    stars = VALUES(stars),
    reviews = VALUES(reviews),
    price = VALUES(price),
    list_price = VALUES(list_price),
    category_id = VALUES(category_id),
    category_name = VALUES(category_name),
    is_best_seller = VALUES(is_best_seller),
    bought_in_last_month = VALUES(bought_in_last_month)
"""


def clean_value(value):
    return value.strip() if value else None


def clean_float(value):
    try:
        return float(value) if value else None
    except (ValueError, TypeError):
        return None


def clean_int(value):
    try:
        return int(float(value)) if value else 0
    except (ValueError, TypeError):
        return 0


def clean_bool(value):
    return value.strip().lower() == "true" if value else False


def ingest_products():
    connection = None

    try:
        if not PRODUCTS_FILE.exists():
            raise FileNotFoundError(
                f"Processed products file not found: {PRODUCTS_FILE}"
            )

        logger.info("Starting MySQL product ingestion")
        logger.info("Input file: %s", PRODUCTS_FILE)
        logger.info("Batch size: %d", BATCH_SIZE)

        connection = get_db_connection()
        cursor = connection.cursor()

        total_inserted = 0
        batch = []

        with PRODUCTS_FILE.open(
            "r",
            encoding="utf-8",
            newline=""
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:
                batch.append((
                    clean_value(row.get("asin")),
                    clean_value(row.get("title")),
                    clean_value(row.get("img_url")),
                    clean_value(row.get("product_url")),
                    clean_float(row.get("stars")),
                    clean_int(row.get("reviews")),
                    clean_float(row.get("price")),
                    clean_float(row.get("list_price")),
                    clean_int(row.get("category_id")),
                    clean_value(row.get("category_name")),
                    clean_bool(row.get("is_best_seller")),
                    clean_int(row.get("bought_in_last_month"))
                ))

                if len(batch) >= BATCH_SIZE:
                    cursor.executemany(INSERT_QUERY, batch)
                    connection.commit()

                    total_inserted += len(batch)

                    logger.info(
                        "Products ingested: %d",
                        total_inserted
                    )

                    batch.clear()

            if batch:
                cursor.executemany(INSERT_QUERY, batch)
                connection.commit()

                total_inserted += len(batch)

                logger.info(
                    "Products ingested: %d",
                    total_inserted
                )

        logger.info(
            "MySQL ingestion completed successfully. Total products: %d",
            total_inserted
        )

    except Exception as e:
        if connection:
            connection.rollback()

        logger.exception("MySQL product ingestion failed")
        raise CustomException(e, sys)

    finally:
        if connection:
            connection.close()
            logger.info("MySQL connection closed")


def main():
    try:
        ingest_products()
    except Exception as e:
        logger.exception("Product ingestion pipeline failed")
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()