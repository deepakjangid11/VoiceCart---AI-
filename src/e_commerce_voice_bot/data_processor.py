import csv
import sys
from pathlib import Path

from e_commerce_voice_bot.exception import CustomException
from e_commerce_voice_bot.logger import get_logger

logger = get_logger(__name__)

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")

PRODUCTS_FILE = RAW_DATA_DIR / "amazon_products.csv"
CATEGORIES_FILE = RAW_DATA_DIR / "amazon_categories.csv"
OUTPUT_FILE = PROCESSED_DATA_DIR / "products_cleaned.csv"


def load_categories():
    try:
        categories = {}

        with CATEGORIES_FILE.open(
            "r",
            encoding="utf-8-sig",
            newline=""
        ) as file:
            reader = csv.DictReader(file)

            for row in reader:
                category_id = row["id"].strip()
                category_name = row["category_name"].strip()

                if category_id:
                    categories[category_id] = category_name

        logger.info("Loaded %d categories", len(categories))

        return categories

    except Exception as e:
        logger.exception("Error while loading categories")
        raise CustomException(e, sys)


def clean_value(value):
    return value.strip() if value else ""


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


def process_products():
    try:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

        categories = load_categories()

        processed_count = 0
        skipped_count = 0

        fieldnames = [
            "asin",
            "title",
            "img_url",
            "product_url",
            "stars",
            "reviews",
            "price",
            "list_price",
            "category_id",
            "category_name",
            "is_best_seller",
            "bought_in_last_month"
        ]

        logger.info("Starting product data processing")
        logger.info("Input file: %s", PRODUCTS_FILE)
        logger.info("Output file: %s", OUTPUT_FILE)

        with PRODUCTS_FILE.open(
            "r",
            encoding="utf-8-sig",
            newline=""
        ) as input_file, OUTPUT_FILE.open(
            "w",
            encoding="utf-8",
            newline=""
        ) as output_file:

            reader = csv.DictReader(input_file)
            writer = csv.DictWriter(
                output_file,
                fieldnames=fieldnames
            )

            writer.writeheader()

            for row in reader:
                asin = clean_value(row.get("asin"))

                if not asin:
                    skipped_count += 1
                    continue

                category_id = clean_value(row.get("category_id"))

                writer.writerow({
                    "asin": asin,
                    "title": clean_value(row.get("title")),
                    "img_url": clean_value(row.get("imgUrl")),
                    "product_url": clean_value(row.get("productURL")),
                    "stars": clean_float(row.get("stars")),
                    "reviews": clean_int(row.get("reviews")),
                    "price": clean_float(row.get("price")),
                    "list_price": clean_float(row.get("listPrice")),
                    "category_id": category_id,
                    "category_name": categories.get(category_id, ""),
                    "is_best_seller": clean_bool(row.get("isBestSeller")),
                    "bought_in_last_month": clean_int(
                        row.get("boughtInLastMonth")
                    )
                })

                processed_count += 1

                if processed_count % 100000 == 0:
                    logger.info(
                        "Products processed: %d",
                        processed_count
                    )

        logger.info("Product processing completed")
        logger.info("Total products processed: %d", processed_count)
        logger.info("Total products skipped: %d", skipped_count)

    except Exception as e:
        logger.exception("Error while processing product data")
        raise CustomException(e, sys)


def main():
    try:
        logger.info("Data processing pipeline started")

        if not PRODUCTS_FILE.exists():
            raise FileNotFoundError(
                f"Products file not found: {PRODUCTS_FILE}"
            )

        if not CATEGORIES_FILE.exists():
            raise FileNotFoundError(
                f"Categories file not found: {CATEGORIES_FILE}"
            )

        process_products()

        logger.info("Data processing pipeline completed")

    except Exception as e:
        logger.exception("Data processing pipeline failed")
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()