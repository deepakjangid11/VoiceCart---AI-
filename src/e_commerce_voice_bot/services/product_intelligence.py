import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from e_commerce_voice_bot.exception import CustomException
from e_commerce_voice_bot.logger import get_logger
from e_commerce_voice_bot.mcp.server import (
    search_products_advanced_tool,
    search_products_tool,
)
from e_commerce_voice_bot.rag import vector_store
from e_commerce_voice_bot.rag.retriever import retrieve_products

logger = get_logger(__name__)


@dataclass
class ProductResult:
    """Normalized product data model across MCP and RAG retrievers."""

    asin: str
    title: str
    price: float | None = None
    score: float = 0.0
    source: str = "mcp"  # "mcp", "rag", or "combined"
    metadata: dict[str, Any] = field(default_factory=dict)


def _parse_price(val: Any) -> float | None:
    """Parse numeric price from various string/numeric representations."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        cleaned = val.replace("$", "").replace(",", "").strip()
        if not cleaned:
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


class ProductIntelligenceService:
    """Unified Product Intelligence Layer orchestrating MCP and RAG retrievals."""

    def __init__(
        self,
        index_file: Path | None = None,
        metadata_db: Path | None = None,
    ) -> None:
        self.index_file = index_file
        self.metadata_db = metadata_db
        self.last_mcp_time: float = 0.0
        self.last_rag_time: float = 0.0
        self.last_total_time: float = 0.0

    def retrieve_structured(
        self,
        query: str,
        limit: int = 10,
        **advanced_filters: Any,
    ) -> list[ProductResult]:
        """Retrieve products using the existing MCP tool interface."""
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty.")

        t0 = time.perf_counter()
        try:
            if advanced_filters:
                raw_json = search_products_advanced_tool(
                    search_term=query.strip(),
                    limit=limit,
                    **advanced_filters,
                )
            else:
                raw_json = search_products_tool(
                    search_term=query.strip(),
                    limit=limit,
                )

            products_data = json.loads(raw_json)
            if not isinstance(products_data, list):
                products_data = [products_data] if products_data else []

            results = []
            for item in products_data:
                if not isinstance(item, dict):
                    continue
                asin = str(item.get("asin") or "").strip()
                if not asin:
                    continue
                title = str(item.get("title") or "").strip()
                price = _parse_price(item.get("price"))
                results.append(
                    ProductResult(
                        asin=asin,
                        title=title,
                        price=price,
                        score=1.0,
                        source="mcp",
                        metadata=item,
                    )
                )

            self.last_mcp_time = time.perf_counter() - t0
            logger.info(
                "Structured retrieval completed | query=%s | count=%d | latency=%.4fs",
                query,
                len(results),
                self.last_mcp_time,
            )
            return results

        except Exception as e:
            self.last_mcp_time = time.perf_counter() - t0
            logger.exception("Structured retrieval failed for query: %s", query)
            raise CustomException(e, sys) from e

    def retrieve_semantic(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[ProductResult]:
        """Retrieve products using the existing semantic RAG retriever."""
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty.")

        t0 = time.perf_counter()
        original_index = vector_store.INDEX_FILE
        original_meta = vector_store.METADATA_DB

        try:
            if self.index_file:
                vector_store.INDEX_FILE = self.index_file
            if self.metadata_db:
                vector_store.METADATA_DB = self.metadata_db

            raw_results = retrieve_products(query.strip(), top_k=top_k)

            results = []
            for item in raw_results:
                meta = item.get("metadata", {})
                asin = str(meta.get("asin") or item.get("id") or "").strip()
                if not asin:
                    continue
                title = str(meta.get("title") or "").strip()
                price = _parse_price(meta.get("price"))
                score = float(item.get("similarity_score", 0.0))

                results.append(
                    ProductResult(
                        asin=asin,
                        title=title,
                        price=price,
                        score=score,
                        source="rag",
                        metadata=meta,
                    )
                )

            self.last_rag_time = time.perf_counter() - t0
            logger.info(
                "Semantic retrieval completed | query=%s | count=%d | latency=%.4fs",
                query,
                len(results),
                self.last_rag_time,
            )
            return results

        except Exception as e:
            self.last_rag_time = time.perf_counter() - t0
            logger.exception("Semantic retrieval failed for query: %s", query)
            raise CustomException(e, sys) from e

        finally:
            vector_store.INDEX_FILE = original_index
            vector_store.METADATA_DB = original_meta

    def retrieve_combined(
        self,
        query: str,
        limit: int = 10,
        top_k: int = 5,
        **advanced_filters: Any,
    ) -> list[ProductResult]:
        """Combine structured (MCP) and semantic (RAG) retrievals with error isolation."""
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty.")

        start_total = time.perf_counter()
        mcp_results: list[ProductResult] = []
        rag_results: list[ProductResult] = []

        mcp_error: Exception | None = None
        rag_error: Exception | None = None

        # 1 & 2. Concurrent structured (MCP) and semantic (RAG) retrievals with error isolation
        with ThreadPoolExecutor(max_workers=2) as executor:
            def _fetch_mcp():
                try:
                    return self.retrieve_structured(
                        query,
                        limit=limit,
                        **advanced_filters,
                    ), None
                except Exception as ex:
                    return [], ex

            def _fetch_rag():
                try:
                    return self.retrieve_semantic(query, top_k=top_k), None
                except Exception as ex:
                    return [], ex

            future_mcp = executor.submit(_fetch_mcp)
            future_rag = executor.submit(_fetch_rag)

            mcp_results, mcp_error = future_mcp.result()
            rag_results, rag_error = future_rag.result()

        if mcp_error is not None:
            logger.warning(
                "MCP retrieval error isolated in combined search: %s",
                mcp_error,
            )

        if rag_error is not None:
            logger.warning(
                "RAG retrieval error isolated in combined search: %s",
                rag_error,
            )

        # If both failed, propagate as CustomException
        if mcp_error is not None and rag_error is not None:
            total_time = time.perf_counter() - start_total
            self.last_total_time = total_time
            logger.error(
                "Both MCP and RAG retrievals failed in combined search for query: %s",
                query,
            )
            combined_msg = (
                f"Combined retrieval failed: MCP error: {mcp_error} | "
                f"RAG error: {rag_error}"
            )
            raise CustomException(combined_msg, sys)

        # 3. Deduplicate and normalize sources
        combined_dict: dict[str, ProductResult] = {}

        for item in mcp_results:
            combined_dict[item.asin] = ProductResult(
                asin=item.asin,
                title=item.title,
                price=item.price,
                score=item.score,
                source="mcp",
                metadata=dict(item.metadata),
            )

        for item in rag_results:
            if item.asin in combined_dict:
                existing = combined_dict[item.asin]
                existing.source = "combined"
                existing.score = max(existing.score, item.score)
                for k, v in item.metadata.items():
                    if k not in existing.metadata or existing.metadata[k] is None:
                        existing.metadata[k] = v
            else:
                combined_dict[item.asin] = ProductResult(
                    asin=item.asin,
                    title=item.title,
                    price=item.price,
                    score=item.score,
                    source="rag",
                    metadata=dict(item.metadata),
                )

        total_time = time.perf_counter() - start_total
        self.last_total_time = total_time

        logger.info(
            "Combined product intelligence completed | "
            "mcp_time=%.4fs | rag_time=%.4fs | total_time=%.4fs | "
            "mcp_count=%d | rag_count=%d | unified_count=%d",
            self.last_mcp_time,
            self.last_rag_time,
            self.last_total_time,
            len(mcp_results),
            len(rag_results),
            len(combined_dict),
        )

        return list(combined_dict.values())
