import os
import sys
import time
from typing import Any

from e_commerce_voice_bot.exception import CustomException
from e_commerce_voice_bot.llm.llm import get_openai_client
from e_commerce_voice_bot.logger import get_logger
from e_commerce_voice_bot.services.product_intelligence import (
    ProductIntelligenceService,
    ProductResult,
)
from e_commerce_voice_bot.services.query_router import (
    QueryRouter,
    QueryRouteResult,
)

logger = get_logger(__name__)

ORCHESTRATOR_SYSTEM_INSTRUCTIONS = (
    "You are VoiceCart AI, a friendly and concise e-commerce voice shopping assistant.\n"
    "Your mission is to provide helpful, natural, and accurate spoken answers to customers.\n\n"
    "CRITICAL RULES:\n"
    "1. Grounding: Answer ONLY using the supplied product context. Never invent prices, ratings, reviews, "
    "features, availability, or product details.\n"
    "2. Unavailable Information: If the context does not contain the answer, or if no products match, "
    "clearly state that the information or product is not available.\n"
    "3. Voice-Friendly & Concise: Write responses designed to be spoken aloud by a voice bot. "
    "Keep sentences clear, direct, and conversational. Avoid bullet lists, Markdown tables, or complex ASCII symbols.\n"
    "4. Explanations: When recommending a product, briefly explain why it fits the customer's request based on the context."
)


def build_compact_context(products: list[ProductResult]) -> str:
    """Build a compact, token-efficient context string from retrieved products."""
    if not products:
        return "No relevant products found in the catalog."

    context_blocks = []
    for rank, p in enumerate(products, start=1):
        price_str = f"${p.price:.2f}" if p.price is not None else "Not available"
        stars = p.metadata.get("stars", "Not available")
        reviews = p.metadata.get("reviews", "Not available")
        category = p.metadata.get("category_name", "Not available")
        bestseller = "Yes" if p.metadata.get("is_best_seller") else "No"

        block = (
            f"[{rank}] Title: {p.title}\n"
            f"    ASIN: {p.asin} | Price: {price_str} | Rating: {stars} ({reviews} reviews)\n"
            f"    Category: {category} | Best Seller: {bestseller} | Source: {p.source}"
        )
        context_blocks.append(block)

    return "\n\n".join(context_blocks)


class ResponseOrchestrator:
    """Unified Response Orchestration Layer connecting Router, Retrieval, and LLM."""

    def __init__(
        self,
        query_router: QueryRouter | None = None,
        product_intelligence: ProductIntelligenceService | None = None,
        llm_client: Any | None = None,
    ) -> None:
        self.query_router = query_router or QueryRouter()
        self.product_intelligence = product_intelligence or ProductIntelligenceService()
        self._llm_client = llm_client

        # Latency metrics
        self.last_router_time: float = 0.0
        self.last_retrieval_time: float = 0.0
        self.last_llm_time: float = 0.0
        self.last_total_time: float = 0.0

    def _get_llm_client(self) -> Any:
        if self._llm_client is not None:
            return self._llm_client
        return get_openai_client()

    def generate_response(
        self,
        query: str,
        limit: int = 5,
        top_k: int = 5,
    ) -> str:
        """Execute end-to-end orchestration: Route -> Retrieve -> Generate Text Response."""
        if not isinstance(query, str) or not query.strip():
            logger.error("Response orchestration rejected: empty or invalid query.")
            raise CustomException("Query string cannot be empty.", sys)

        cleaned_query = query.strip()
        start_total = time.perf_counter()

        # Step 1: Call Query Router
        t0_router = time.perf_counter()
        try:
            route_result: QueryRouteResult = self.query_router.route_query(cleaned_query)
        except Exception as e:
            logger.exception("Router failure during response orchestration: %s", e)
            raise CustomException(f"Router failure: {e}", sys) from e
        finally:
            self.last_router_time = time.perf_counter() - t0_router

        # Step 2: Execute selected retrieval strategy
        t0_retrieval = time.perf_counter()
        products: list[ProductResult] = []
        try:
            if route_result.route == "mcp":
                products = self.product_intelligence.retrieve_structured(
                    cleaned_query,
                    limit=limit,
                )
            elif route_result.route == "rag":
                products = self.product_intelligence.retrieve_semantic(
                    cleaned_query,
                    top_k=top_k,
                )
            elif route_result.route == "both":
                products = self.product_intelligence.retrieve_combined(
                    cleaned_query,
                    limit=limit,
                    top_k=top_k,
                )
            else:
                raise ValueError(f"Unrecognized route: {route_result.route}")
        except Exception as e:
            logger.exception("Retrieval failure during response orchestration: %s", e)
            raise CustomException(f"Retrieval failure: {e}", sys) from e
        finally:
            self.last_retrieval_time = time.perf_counter() - t0_retrieval

        # Step 3: Build compact context
        product_context = build_compact_context(products)

        # Step 4: Send context + query to LLM
        t0_llm = time.perf_counter()
        try:
            client = self._get_llm_client()
            model = os.getenv("OPENAI_MODEL", "gpt-5.5")

            if hasattr(client, "responses") and hasattr(client.responses, "create"):
                response = client.responses.create(
                    model=model,
                    instructions=ORCHESTRATOR_SYSTEM_INSTRUCTIONS,
                    input=(
                        f"Customer question: {cleaned_query}\n\n"
                        f"Retrieved product context:\n{product_context}"
                    ),
                )
                final_response = getattr(response, "output_text", None) or str(response)
            elif hasattr(client, "chat") and hasattr(client.chat, "completions"):
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": ORCHESTRATOR_SYSTEM_INSTRUCTIONS},
                        {
                            "role": "user",
                            "content": (
                                f"Customer question: {cleaned_query}\n\n"
                                f"Retrieved product context:\n{product_context}"
                            ),
                        },
                    ],
                )
                final_response = response.choices[0].message.content
            else:
                raise ValueError("Unsupported OpenAI client interface.")

            final_text = str(final_response).strip()

        except Exception as e:
            logger.exception("LLM failure during response orchestration: %s", e)
            raise CustomException(f"LLM failure: {e}", sys) from e
        finally:
            self.last_llm_time = time.perf_counter() - t0_llm
            self.last_total_time = time.perf_counter() - start_total

        # Step 5: Structured timing log
        logger.info(
            "Response orchestration completed | "
            "router_time=%.4fs | retrieval_time=%.4fs | llm_time=%.4fs | total_time=%.4fs | "
            "route=%s | products_retrieved=%d | query=%s",
            self.last_router_time,
            self.last_retrieval_time,
            self.last_llm_time,
            self.last_total_time,
            route_result.route,
            len(products),
            cleaned_query,
        )

        return final_text
