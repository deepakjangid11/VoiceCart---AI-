import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Literal

from e_commerce_voice_bot.exception import CustomException
from e_commerce_voice_bot.llm.llm import get_openai_client
from e_commerce_voice_bot.logger import get_logger

logger = get_logger(__name__)

RouteType = Literal["mcp", "rag", "both"]
VALID_ROUTES: set[str] = {"mcp", "rag", "both"}

ROUTER_SYSTEM_INSTRUCTIONS = (
    "You are the intent router for VoiceCart AI, an e-commerce platform.\n"
    "Your task is to analyze customer queries and decide the optimal retrieval strategy.\n"
    "Choose exactly one of the following three routes:\n\n"
    "1. 'mcp': Use when the query asks for live or structured product data, such as:\n"
    "   - Product pricing, price limits (e.g., 'under $50')\n"
    "   - Bestsellers and category listings\n"
    "   - Customer reviews and star ratings\n"
    "   - Direct ASIN or product attribute lookup\n\n"
    "2. 'rag': Use when the query is semantic, advisory, or knowledge-oriented, such as:\n"
    "   - Guides, manuals, and instructional queries (e.g., 'guide for birds')\n"
    "   - Subject-matter suitability (e.g., 'suitable for learning photography')\n"
    "   - Conceptual product recommendations without strict numeric filters\n\n"
    "3. 'both': Use ONLY when the query requires BOTH live structured filtering AND semantic reasoning:\n"
    "   - Example: 'Find laptops under $1000 and explain which one is best for programming.'\n\n"
    "CRITICAL OUTPUT RULES:\n"
    "- Return ONLY a valid JSON object matching the schema below.\n"
    "- Do NOT output any markdown code blocks, prefixes, or commentary.\n"
    "- The 'route' field MUST be strictly one of: 'mcp', 'rag', 'both'.\n"
    "- Keep 'reason' concise (1-2 sentences).\n\n"
    "JSON Schema:\n"
    "{\n"
    '  "route": "mcp" | "rag" | "both",\n'
    '  "reason": "concise explanation for route selection"\n'
    "}"
)


@dataclass
class QueryRouteResult:
    """Strict result model for query intent routing."""

    route: RouteType
    query: str
    reason: str


class QueryRouter:
    """Lightweight LLM-based query router classifying queries into mcp, rag, or both."""

    def __init__(self, client: Any | None = None) -> None:
        self._client = client
        self.last_router_time: float = 0.0

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        return get_openai_client()

    def route_query(self, query: str) -> QueryRouteResult:
        """Classify user query into 'mcp', 'rag', or 'both' using the existing LLM."""
        if not isinstance(query, str) or not query.strip():
            logger.error("Query routing rejected: empty or non-string query provided.")
            raise CustomException("Query string cannot be empty.", sys)

        cleaned_query = query.strip()
        t0 = time.perf_counter()

        try:
            client = self._get_client()
            model = os.getenv("OPENAI_MODEL", "gpt-5.5")

            logger.info("Routing query with LLM | model=%s | query=%s", model, cleaned_query)

            # Call existing LLM via responses API (or fallback if client signature differs)
            if hasattr(client, "responses") and hasattr(client.responses, "create"):
                response = client.responses.create(
                    model=model,
                    instructions=ROUTER_SYSTEM_INSTRUCTIONS,
                    input=f"Customer query: {cleaned_query}",
                )
                raw_text = getattr(response, "output_text", None) or str(response)
            elif hasattr(client, "chat") and hasattr(client.chat, "completions"):
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": ROUTER_SYSTEM_INSTRUCTIONS},
                        {"role": "user", "content": cleaned_query},
                    ],
                )
                raw_text = response.choices[0].message.content
            else:
                raise ValueError("Unsupported OpenAI client interface.")

            # Clean and parse JSON
            cleaned_text = raw_text.strip()
            if cleaned_text.startswith("```"):
                lines = cleaned_text.splitlines()
                if lines and lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                cleaned_text = "\n".join(lines).strip()

            try:
                data = json.loads(cleaned_text)
            except json.JSONDecodeError as err:
                logger.error("Failed to parse LLM router JSON output: %s", raw_text)
                raise ValueError(
                    f"Malformed LLM router output, invalid JSON: {raw_text}"
                ) from err

            if not isinstance(data, dict):
                logger.error("LLM router output is not a JSON object: %s", type(data).__name__)
                raise ValueError(
                    f"Malformed LLM router output, expected dict, got: {type(data).__name__}"
                )

            route_val = str(data.get("route", "")).strip().lower()
            if route_val not in VALID_ROUTES:
                logger.error("Invalid route returned by LLM: %s", route_val)
                raise ValueError(
                    f"Invalid route '{route_val}'. Route must be one of: {sorted(VALID_ROUTES)}"
                )

            reason_val = str(data.get("reason", "")).strip()
            if not reason_val:
                reason_val = f"Classified query to {route_val} route."

            self.last_router_time = time.perf_counter() - t0
            logger.info(
                "Query routed successfully | route=%s | llm_router_time=%.4fs | query=%s",
                route_val,
                self.last_router_time,
                cleaned_query,
            )

            return QueryRouteResult(
                route=route_val,  # type: ignore[arg-type]
                query=cleaned_query,
                reason=reason_val,
            )

        except Exception as e:
            self.last_router_time = time.perf_counter() - t0
            logger.exception("Query routing failed | query=%s | error=%s", cleaned_query, e)
            if isinstance(e, CustomException):
                raise
            raise CustomException(e, sys) from e
