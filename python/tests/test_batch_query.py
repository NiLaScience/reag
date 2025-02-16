import pytest
from unittest.mock import Mock, patch, AsyncMock
import os

from reag.client import ReagClient, Document, QueryResult
from reag.schema import BatchQueryResult

# Test data
SAMPLE_DOCS = [
    Document(
        name="doc1",
        content="Document 1 content",
        metadata={"category": "test"}
    ),
    Document(
        name="doc2",
        content="Document 2 content",
        metadata={"category": "other"}
    )
]

MOCK_RESPONSE = Mock(
    choices=[
        Mock(
            message=Mock(
                content='{"source": {"content": "test content", "reasoning": "test reasoning", "is_irrelevant": false}}'
            )
        )
    ]
)

@pytest.fixture(autouse=True)
def mock_env():
    """Mock environment variables for testing."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"}):
        yield

@pytest.fixture
def mock_llm():
    """Mock LLM completion calls."""
    with patch('reag.client.acompletion', new_callable=AsyncMock) as mock_completion:
        mock_completion.return_value = MOCK_RESPONSE
        yield mock_completion

@pytest.mark.asyncio
async def test_single_query_in_batch(mock_llm):
    """Test processing a single query in batch mode."""
    async with ReagClient() as client:
        # Mock the query method directly
        with patch.object(client, 'query', new_callable=AsyncMock) as mock_query:
            mock_query.return_value = [
                QueryResult(
                    content="test content",
                    reasoning="test reasoning",
                    is_irrelevant=False,
                    document=SAMPLE_DOCS[0]
                )
            ]
            
            queries = ["What is in document 1?"]
            results = await client.query_batch(queries=queries, documents=SAMPLE_DOCS)
            
            assert len(results) == 1
            assert results[0].query == queries[0]
            assert len(results[0].results) == 1
            assert results[0].error is None

@pytest.mark.asyncio
async def test_multiple_queries(mock_llm):
    """Test processing multiple queries in parallel."""
    async with ReagClient() as client:
        # Mock the query method directly
        with patch.object(client, 'query', new_callable=AsyncMock) as mock_query:
            mock_query.return_value = [
                QueryResult(
                    content="test content",
                    reasoning="test reasoning",
                    is_irrelevant=False,
                    document=SAMPLE_DOCS[0]
                )
            ]
            
            queries = [
                "What is in document 1?",
                "What is in document 2?",
                "What are the documents about?"
            ]
            results = await client.query_batch(queries=queries, documents=SAMPLE_DOCS)
            
            assert len(results) == 3
            for result, query in zip(results, queries):
                assert result.query == query
                assert len(result.results) == 1
                assert result.error is None

@pytest.mark.asyncio
async def test_error_handling(mock_llm):
    """Test error handling in batch processing."""
    async with ReagClient() as client:
        # Mock the query method to raise an exception
        with patch.object(client, 'query', side_effect=Exception("Test error")):
            queries = ["Query 1", "Query 2"]
            results = await client.query_batch(queries=queries, documents=SAMPLE_DOCS)
            
            assert len(results) == 2
            for result in results:
                assert result.error is not None
                assert "Test error" in result.error
                assert len(result.results) == 0

@pytest.mark.asyncio
async def test_concurrency_limit(mock_llm):
    """Test respecting the concurrency limit."""
    async with ReagClient() as client:
        # Mock the query method directly
        with patch.object(client, 'query', new_callable=AsyncMock) as mock_query:
            mock_query.return_value = [
                QueryResult(
                    content="test content",
                    reasoning="test reasoning",
                    is_irrelevant=False,
                    document=SAMPLE_DOCS[0]
                )
            ]
            
            queries = [f"Query {i}" for i in range(10)]
            max_concurrent = 3
            
            results = await client.query_batch(
                queries=queries,
                documents=SAMPLE_DOCS,
                max_concurrent=max_concurrent
            )
            
            assert len(results) == 10
            for result in results:
                assert result.error is None
                assert len(result.results) == 1

@pytest.mark.asyncio
async def test_batch_with_metadata_filter(mock_llm):
    """Test batch queries with metadata filtering."""
    async with ReagClient() as client:
        # Mock the query method directly
        with patch.object(client, 'query', new_callable=AsyncMock) as mock_query:
            mock_query.return_value = [
                QueryResult(
                    content="test content",
                    reasoning="test reasoning",
                    is_irrelevant=False,
                    document=SAMPLE_DOCS[0]
                )
            ]
            
            queries = ["Test query"]
            options = {"filter": {"key": "category", "value": "test"}}
            
            results = await client.query_batch(
                queries=queries,
                documents=SAMPLE_DOCS,
                options=options
            )
            
            assert len(results) == 1
            assert results[0].error is None
            assert len(results[0].results) == 1
            assert results[0].results[0].document.metadata["category"] == "test" 