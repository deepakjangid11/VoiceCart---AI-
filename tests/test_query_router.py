import json
from unittest.mock import MagicMock

import pytest

from e_commerce_voice_bot.exception import CustomException
from e_commerce_voice_bot.services.query_router import (
    QueryRouter,
    QueryRouteResult,
)


def _build_mock_client(output_text: str):
    """Create a mock OpenAI client returning specified output text via responses.create."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.output_text = output_text
    mock_client.responses.create.return_value = mock_response
    return mock_client


def test_mcp_classification():
    """Validate that structured queries (price, bestseller, reviews) are routed to 'mcp'."""
    query = "Show me products under $50"
    mock_output = json.dumps(
        {
            "route": "mcp",
            "reason": "Structured price filtering requested under $50 threshold.",
        }
    )
    router = QueryRouter(client=_build_mock_client(mock_output))
    result = router.route_query(query)

    assert isinstance(result, QueryRouteResult)
    assert result.route == "mcp"
    assert result.query == query
    assert "Structured price filtering" in result.reason
    assert router.last_router_time >= 0


def test_rag_classification():
    """Validate that semantic/knowledge queries (guides, suitability) are routed to 'rag'."""
    query = "I need a good guide for birds"
    mock_output = json.dumps(
        {
            "route": "rag",
            "reason": "Semantic informational query looking for bird identification guides.",
        }
    )
    router = QueryRouter(client=_build_mock_client(mock_output))
    result = router.route_query(query)

    assert isinstance(result, QueryRouteResult)
    assert result.route == "rag"
    assert result.query == query
    assert "Semantic informational" in result.reason
    assert router.last_router_time >= 0


def test_both_classification():
    """Validate that hybrid queries requiring both live data and advice route to 'both'."""
    query = (
        "Find me highly rated laptops under $1000 and explain which one is best for programming."
    )
    mock_output = json.dumps(
        {
            "route": "both",
            "reason": "Requires structured price/rating filtering and semantic programming suitability reasoning.",
        }
    )
    router = QueryRouter(client=_build_mock_client(mock_output))
    result = router.route_query(query)

    assert isinstance(result, QueryRouteResult)
    assert result.route == "both"
    assert result.query == query
    assert "both" in result.route
    assert router.last_router_time >= 0


def test_structured_result_validation():
    """Validate the schema and integrity of QueryRouteResult."""
    query = "What is the price of this product?"
    mock_output = json.dumps(
        {
            "route": "mcp",
            "reason": "Direct product price attribute lookup.",
        }
    )
    router = QueryRouter(client=_build_mock_client(mock_output))
    result = router.route_query(query)

    assert hasattr(result, "route")
    assert hasattr(result, "query")
    assert hasattr(result, "reason")
    assert result.route in {"mcp", "rag", "both"}
    assert isinstance(result.query, str)
    assert isinstance(result.reason, str)
    assert len(result.reason) > 0


def test_markdown_code_block_parsing():
    """Validate that LLM responses wrapped in markdown code blocks are safely parsed."""
    query = "How many reviews does this product have?"
    mock_output = "```json\n{\"route\": \"mcp\", \"reason\": \"Review count lookup\"}\n```"
    router = QueryRouter(client=_build_mock_client(mock_output))
    result = router.route_query(query)

    assert result.route == "mcp"
    assert result.reason == "Review count lookup"


def test_chat_completions_fallback():
    """Validate fallback support for chat.completions client interface."""
    query = "Which product is suitable for learning photography?"
    mock_client = MagicMock()
    # Remove responses attribute to force chat.completions branch
    del mock_client.responses
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps(
        {"route": "rag", "reason": "Photography learning suitability query"}
    )
    mock_client.chat.completions.create.return_value.choices = [mock_choice]

    router = QueryRouter(client=mock_client)
    result = router.route_query(query)

    assert result.route == "rag"
    assert "Photography learning" in result.reason


def test_empty_query_rejection():
    """Validate that empty queries are rejected with CustomException."""
    router = QueryRouter(client=_build_mock_client("{}"))

    with pytest.raises(CustomException):
        router.route_query("")

    with pytest.raises(CustomException):
        router.route_query("   ")


def test_invalid_route_rejection():
    """Validate that hallucinated or invalid routes trigger a CustomException."""
    query = "Tell me about this item"
    mock_output = json.dumps({"route": "google_search", "reason": "invalid route"})
    router = QueryRouter(client=_build_mock_client(mock_output))

    with pytest.raises(CustomException) as exc_info:
        router.route_query(query)

    assert "Invalid route 'google_search'" in str(exc_info.value)


def test_malformed_json_rejection():
    """Validate that non-JSON output triggers a CustomException."""
    query = "What products are best sellers?"
    mock_output = "I think you should use MCP for this query because it is about best sellers."
    router = QueryRouter(client=_build_mock_client(mock_output))

    with pytest.raises(CustomException) as exc_info:
        router.route_query(query)

    assert "Malformed LLM router output" in str(exc_info.value)
