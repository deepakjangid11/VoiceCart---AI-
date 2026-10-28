
import os

from dotenv import load_dotenv
from openai import OpenAI

from e_commerce_voice_bot.logger import get_logger
from e_commerce_voice_bot.rag.retriever import retrieve_products

logger = get_logger(__name__)
load_dotenv()


def get_openai_client() -> OpenAI:
    """Create an OpenAI client using the configured API key."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is missing. Add it to your .env file."
        )

    return OpenAI(api_key=api_key)


def generate_shopping_response(
    query: str,
    top_k: int = 5,
) -> str:
    """Generate a shopping answer grounded in retrieved products."""
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    try:
        products = retrieve_products(query, top_k=top_k)

        if not products:
            return "Sorry, I couldn't find relevant products for your query."

        product_context = "\n\n".join(
            (
                f"Product: {item.get('metadata', {}).get('title', 'Unknown')}\n"
                f"Category: {item.get('metadata', {}).get('category_name', 'Unknown')}\n"
                f"Price: {item.get('metadata', {}).get('price', 'Not available')}\n"
                f"Rating: {item.get('metadata', {}).get('stars', 'Not available')}"
            )
            for item in products
        )

        client = get_openai_client()
        model = os.getenv("OPENAI_MODEL", "gpt-5.5")

        response = client.responses.create(
            model=model,
            instructions=(
                "You are VoiceCart AI, a helpful e-commerce shopping assistant. "
                "Recommend products using only the supplied product context. "
                "Never invent prices, ratings, product features, or availability. "
                "If the context does not answer a question, clearly say so. "
                "Be concise, friendly, and explain why a product may fit the request."
            ),
            input=(
                f"Customer question: {query}\n\n"
                f"Retrieved product context:\n{product_context}"
            ),
        )

        answer = response.output_text
        logger.info("Generated shopping response successfully.")
        return answer

    except Exception:
        logger.exception("Failed to generate shopping response.")
        raise
