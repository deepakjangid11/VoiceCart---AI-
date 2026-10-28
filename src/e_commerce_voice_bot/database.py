
import os
import sys

import mysql.connector
from dotenv import load_dotenv

from e_commerce_voice_bot.exception import CustomException
from e_commerce_voice_bot.logger import get_logger

load_dotenv()

logger = get_logger(__name__)


def get_db_connection():
    """Create and return a MySQL database connection."""
    try:
        connection = mysql.connector.connect(
            host=os.getenv("DB_HOST"),
            port=int(os.getenv("DB_PORT", "3306")),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_NAME"),
        )

        logger.info("MySQL database connection established.")
        return connection

    except Exception as e:
        logger.exception("Failed to connect to MySQL database.")
        raise CustomException(e, sys) from e
