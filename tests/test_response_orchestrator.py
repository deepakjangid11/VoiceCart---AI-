from unittest.mock import MagicMock

import pytest

from e_commerce_voice_bot.exception import CustomException
from e_commerce_voice_bot.services.product_intelligence import ProductResult
from e_commerce_voice_bot.services.query_router import QueryRouteResult
from e_commerce_voice_bot.services.response_orchestrator import (
    ResponseOrchestrator,
    build_compact_context,
)


@pytest.fixture
def sample_products():
    """Sample normalized product results for mocking."""
    return [
        ProductResult(
            asin="B001TEST01",
            title="Premium Noise Cancelling Headphones",
            price=199.99,
            score=0.92,
            source="combined",
            metadata={
                "stars": 4.6,
                "reviews": 1250,
                "category_name": "Electronics",
                "is_best_seller": True,
            },
        ),
        ProductResult(
            asin="B002TEST02",
            title="Budget Wireless Earbuds",
            price=29.99,
            score=0.85,
            source="mcp",
            metadata={
                "stars": 4.2,
                "reviews": 320,
                "category_name": "Electronics",
                "is_best_seller": False,
            },
        ),
    ]


@pytest.fixture
def mock_router():
    return MagicMock()


@pytest.fixture
def mock_intelligence(sample_products):
    intel = MagicMock()
    intel.retrieve_structured.return_value = [sample_products[1]]
    intel.retrieve_semantic.return_value = [sample_products[0]]
    intel.retrieve_combined.return_value = sample_products
    return intel


@pytest.fixture
def mock_llm_client():
    client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.output_text = (
        "I recommend the Premium Noise Cancelling Headphones for $199.99 with a 4.6 rating."
    )
    client.responses.create.return_value = mock_resp
    return client


def test_build_compact_context(sample_products):
    """Validate context string formatting from product results."""
    context = build_compact_context(sample_products)
    assert "Premium Noise Cancelling Headphones" in context
    assert "$199.99" in context
    assert "ASIN: B001TEST01" in context
    assert "Source: combined" in context

    empty_context = build_compact_context([])
    assert "No relevant products found" in empty_context


def test_mcp_route_execution(mock_router, mock_intelligence, mock_llm_client):
    """Validate MCP route invokes retrieve_structured and generates response."""
    query = "Show me products under $50"
    mock_router.route_query.return_value = QueryRouteResult(
        route="mcp",
        query=query,
        reason="Price filter query",
    )

    orchestrator = ResponseOrchestrator(
        query_router=mock_router,
        product_intelligence=mock_intelligence,
        llm_client=mock_llm_client,
    )

    response = orchestrator.generate_response(query)

    mock_router.route_query.assert_called_once_with(query)
    mock_intelligence.retrieve_structured.assert_called_once_with(query, limit=5)
    mock_intelligence.retrieve_semantic.assert_not_called()
    mock_intelligence.retrieve_combined.assert_not_called()
    mock_llm_client.responses.create.assert_called_once()
    assert "Premium Noise Cancelling Headphones" in response


def test_rag_route_execution(mock_router, mock_intelligence, mock_llm_client):
    """Validate RAG route invokes retrieve_semantic and generates response."""
    query = "I need a good guide for birds"
    mock_router.route_query.return_value = QueryRouteResult(
        route="rag",
        query=query,
        reason="Educational guide query",
    )

    orchestrator = ResponseOrchestrator(
        query_router=mock_router,
        product_intelligence=mock_intelligence,
        llm_client=mock_llm_client,
    )

    response = orchestrator.generate_response(query)

    mock_router.route_query.assert_called_once_with(query)
    mock_intelligence.retrieve_semantic.assert_called_once_with(query, top_k=5)
    mock_intelligence.retrieve_structured.assert_not_called()
    mock_intelligence.retrieve_combined.assert_not_called()
    assert len(response) > 0


def test_both_route_execution(mock_router, mock_intelligence, mock_llm_client):
    """Validate BOTH route invokes retrieve_combined (concurrent retrieval)."""
    query = "Find laptops under $1000 and explain which is best for programming"
    mock_router.route_query.return_value = QueryRouteResult(
        route="both",
        query=query,
        reason="Hybrid price filter and suitability",
    )

    orchestrator = ResponseOrchestrator(
        query_router=mock_router,
        product_intelligence=mock_intelligence,
        llm_client=mock_llm_client,
    )

    response = orchestrator.generate_response(query)

    mock_router.route_query.assert_called_once_with(query)
    mock_intelligence.retrieve_combined.assert_called_once_with(query, limit=5, top_k=5)
    mock_intelligence.retrieve_structured.assert_not_called()
    mock_intelligence.retrieve_semantic.assert_not_called()
    assert len(response) > 0


def test_final_response_generation(mock_router, mock_intelligence, mock_llm_client):
    """Validate the exact output text returned by the LLM is returned by the orchestrator."""
    query = "Find earbuds"
    mock_router.route_query.return_value = QueryRouteResult(
        route="mcp",
        query=query,
        reason="Product search",
    )
    expected_text = "Here is the top-rated Budget Wireless Earbuds for $29.99."
    mock_llm_client.responses.create.return_value.output_text = expected_text

    orchestrator = ResponseOrchestrator(
        query_router=mock_router,
        product_intelligence=mock_intelligence,
        llm_client=mock_llm_client,
    )

    result = orchestrator.generate_response(query)
    assert result == expected_text


def test_empty_query_rejection(mock_router, mock_intelligence, mock_llm_client):
    """Validate empty/whitespace query raises CustomException."""
    orchestrator = ResponseOrchestrator(
        query_router=mock_router,
        product_intelligence=mock_intelligence,
        llm_client=mock_llm_client,
    )

    with pytest.raises(CustomException):
        orchestrator.generate_response("")

    with pytest.raises(CustomException):
        orchestrator.generate_response("   ")


def test_router_failure_handling(mock_router, mock_intelligence, mock_llm_client):
    """Validate router failures are captured as CustomException."""
    mock_router.route_query.side_effect = RuntimeError("Router timeout")

    orchestrator = ResponseOrchestrator(
        query_router=mock_router,
        product_intelligence=mock_intelligence,
        llm_client=mock_llm_client,
    )

    with pytest.raises(CustomException) as exc_info:
        orchestrator.generate_response("Valid query")

    assert "Router failure" in str(exc_info.value)


def test_retrieval_failure_handling(mock_router, mock_intelligence, mock_llm_client):
    """Validate retrieval failures are captured as CustomException."""
    mock_router.route_query.return_value = QueryRouteResult(
        route="mcp",
        query="test",
        reason="test",
    )
    mock_intelligence.retrieve_structured.side_effect = RuntimeError("Database offline")

    orchestrator = ResponseOrchestrator(
        query_router=mock_router,
        product_intelligence=mock_intelligence,
        llm_client=mock_llm_client,
    )

    with pytest.raises(CustomException) as exc_info:
        orchestrator.generate_response("test")

    assert "Retrieval failure" in str(exc_info.value)


def test_llm_failure_handling(mock_router, mock_intelligence, mock_llm_client):
    """Validate LLM service failures are captured as CustomException."""
    mock_router.route_query.return_value = QueryRouteResult(
        route="rag",
        query="test",
        reason="test",
    )
    mock_llm_client.responses.create.side_effect = RuntimeError("OpenAI rate limit")

    orchestrator = ResponseOrchestrator(
        query_router=mock_router,
        product_intelligence=mock_intelligence,
        llm_client=mock_llm_client,
    )

    with pytest.raises(CustomException) as exc_info:
        orchestrator.generate_response("test")

    assert "LLM failure" in str(exc_info.value)


def test_timing_and_logging_metrics(mock_router, mock_intelligence, mock_llm_client):
    """Validate all latency metrics are accurately recorded."""
    mock_router.route_query.return_value = QueryRouteResult(
        route="both",
        query="laptop",
        reason="hybrid",
    )

    orchestrator = ResponseOrchestrator(
        query_router=mock_router,
        product_intelligence=mock_intelligence,
        llm_client=mock_llm_client,
    )

    orchestrator.generate_response("laptop")

    assert orchestrator.last_router_time >= 0
    assert orchestrator.last_retrieval_time >= 0
    assert orchestrator.last_llm_time >= 0
    assert orchestrator.last_total_time >= orchestrator.last_llm_time
