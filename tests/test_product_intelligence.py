from pathlib import Path

import pytest

from e_commerce_voice_bot.exception import CustomException
from e_commerce_voice_bot.services.product_intelligence import (
    ProductIntelligenceService,
    ProductResult,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_INDEX_FILE = PROJECT_ROOT / "data" / "vector_store_test" / "test_products.index"
TEST_METADATA_DB = PROJECT_ROOT / "data" / "vector_store_test" / "test_metadata.db"


@pytest.fixture
def intelligence_service():
    """Create a ProductIntelligenceService configured with the small test vector store."""
    return ProductIntelligenceService(
        index_file=TEST_INDEX_FILE,
        metadata_db=TEST_METADATA_DB,
    )


def test_structured_retrieval(intelligence_service):
    """Validate structured retrieval via the MCP tool layer."""
    results = intelligence_service.retrieve_structured("iphone", limit=3)
    assert isinstance(results, list)
    if results:
        for r in results:
            assert isinstance(r, ProductResult)
            assert r.asin
            assert r.title
            assert r.source == "mcp"
            assert r.score == 1.0


def test_rag_retrieval(intelligence_service):
    """Validate semantic retrieval via the RAG layer using the test index."""
    results = intelligence_service.retrieve_semantic("bird guide", top_k=3)
    assert len(results) > 0
    top_match = results[0]
    assert top_match.asin == "0007268149"
    assert "Collins Bird Guide" in top_match.title
    assert top_match.source == "rag"
    assert top_match.score > 0.0
    assert intelligence_service.last_rag_time > 0


def test_combined_retrieval(intelligence_service):
    """Validate combined retrieval latency tracking and result formatting."""
    results = intelligence_service.retrieve_combined("bird guide", limit=2, top_k=2)
    assert isinstance(results, list)
    assert len(results) > 0
    assert intelligence_service.last_mcp_time >= 0
    assert intelligence_service.last_rag_time > 0
    assert intelligence_service.last_total_time > 0


def test_normalized_result_fields(intelligence_service):
    """Validate that every normalized ProductResult conforms to required schema."""
    results = intelligence_service.retrieve_semantic("building", top_k=2)
    assert len(results) > 0
    for r in results:
        assert isinstance(r.asin, str)
        assert isinstance(r.title, str)
        assert r.price is None or isinstance(r.price, float)
        assert isinstance(r.score, float)
        assert r.source in {"mcp", "rag", "combined"}


def test_duplicate_asin_removal_and_source_tracking(intelligence_service, monkeypatch):
    """Validate that overlapping ASINs between MCP and RAG are deduplicated and marked 'combined'."""
    mock_mcp_results = [
        ProductResult(
            asin="B000TEST01",
            title="Shared Product One",
            price=19.99,
            score=1.0,
            source="mcp",
            metadata={"category_name": "Electronics"},
        ),
        ProductResult(
            asin="B000TEST02",
            title="Unique MCP Product",
            price=29.99,
            score=1.0,
            source="mcp",
            metadata={"category_name": "Home"},
        ),
    ]

    mock_rag_results = [
        ProductResult(
            asin="B000TEST01",
            title="Shared Product One Semantic",
            price=19.99,
            score=0.89,
            source="rag",
            metadata={"reviews": 150},
        ),
        ProductResult(
            asin="B000TEST03",
            title="Unique RAG Product",
            price=49.99,
            score=0.75,
            source="rag",
            metadata={"category_name": "Books"},
        ),
    ]

    monkeypatch.setattr(
        intelligence_service,
        "retrieve_structured",
        lambda query, limit=10, **kw: mock_mcp_results,
    )
    monkeypatch.setattr(
        intelligence_service,
        "retrieve_semantic",
        lambda query, top_k=5: mock_rag_results,
    )

    combined = intelligence_service.retrieve_combined("test query")

    # Should contain exactly 3 unique products: TEST01, TEST02, TEST03
    assert len(combined) == 3
    asins = {r.asin: r for r in combined}

    # B000TEST01 was in both -> must be 'combined'
    assert asins["B000TEST01"].source == "combined"
    assert asins["B000TEST01"].metadata.get("category_name") == "Electronics"
    assert asins["B000TEST01"].metadata.get("reviews") == 150

    # B000TEST02 was only in MCP -> must be 'mcp'
    assert asins["B000TEST02"].source == "mcp"

    # B000TEST03 was only in RAG -> must be 'rag'
    assert asins["B000TEST03"].source == "rag"


def test_mcp_failure_isolation(intelligence_service, monkeypatch):
    """Validate that an MCP database/network failure does not fail RAG results."""
    def fail_mcp(*args, **kwargs):
        raise RuntimeError("MySQL connection timed out")

    monkeypatch.setattr(intelligence_service, "retrieve_structured", fail_mcp)

    # Call combined search: RAG should succeed even though MCP raised
    results = intelligence_service.retrieve_combined("bird guide", top_k=2)

    assert len(results) > 0
    for r in results:
        assert r.source == "rag"


def test_rag_failure_isolation(intelligence_service, monkeypatch):
    """Validate that a RAG/FAISS failure does not fail MCP results."""
    def fail_rag(*args, **kwargs):
        raise RuntimeError("FAISS index inaccessible")

    mock_mcp = [
        ProductResult(
            asin="B000MOCK01",
            title="Fallback MCP Product",
            price=12.50,
            score=1.0,
            source="mcp",
        )
    ]

    monkeypatch.setattr(intelligence_service, "retrieve_semantic", fail_rag)
    monkeypatch.setattr(
        intelligence_service,
        "retrieve_structured",
        lambda *args, **kw: mock_mcp,
    )

    results = intelligence_service.retrieve_combined("mock query")

    assert len(results) == 1
    assert results[0].asin == "B000MOCK01"
    assert results[0].source == "mcp"


def test_both_failures_raise_custom_exception(intelligence_service, monkeypatch):
    """Validate that if both MCP and RAG fail, a CustomException is raised."""
    def fail_mcp(*args, **kwargs):
        raise RuntimeError("MCP failed")

    def fail_rag(*args, **kwargs):
        raise RuntimeError("RAG failed")

    monkeypatch.setattr(intelligence_service, "retrieve_structured", fail_mcp)
    monkeypatch.setattr(intelligence_service, "retrieve_semantic", fail_rag)

    with pytest.raises(CustomException):
        intelligence_service.retrieve_combined("query that causes total failure")
