import pytest
from unittest.mock import Mock, patch, AsyncMock
from typing import List
import os
import json

from reag.client import ReagClient, Document, QueryResult, RagConfig
from reag.schema import MetadataFilter
from reag.ranking import DocumentRanker

# Test data
SAMPLE_DOCS = [
    Document(
        name="Superagent",
        content="Superagent is a workspace for AI-agents that learn, perform work, and collaborate.",
        metadata={"url": "https://superagent.sh", "source": "web"}
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
async def test_query_with_documents(mock_llm):
    """Test querying documents returns expected results."""
    async with ReagClient() as client:
        response = await client.query("What is Superagent?", documents=SAMPLE_DOCS)
        assert len(response) == 1
        assert response[0].content == "test content"
        assert response[0].reasoning == "test reasoning"
        assert response[0].document == SAMPLE_DOCS[0]

@pytest.mark.asyncio
async def test_query_with_metadata_filter(mock_llm):
    """Test query with metadata filtering."""
    async with ReagClient() as client:
        response = await client.query(
            "test query",
            documents=SAMPLE_DOCS,
            options={"filter": {"key": "source", "value": "web"}}
        )
        assert len(response) == 1
        assert response[0].document == SAMPLE_DOCS[0]

@pytest.mark.asyncio
async def test_query_with_integer_filter(mock_llm):
    """Test query with integer metadata filtering."""
    docs = [Document(
        name="Test Doc",
        content="Test content",
        metadata={"priority": 1}
    )]
    
    async with ReagClient() as client:
        response = await client.query(
            "test query",
            documents=docs,
            options={"filter": {"key": "priority", "value": 1}}
        )
        assert len(response) == 1
        assert response[0].document == docs[0]

@pytest.mark.asyncio
async def test_query_with_ranking(mock_llm):
    """Test query with document ranking enabled."""
    # Create sample ranked response
    ranked_response = {
        "rankings": [
            {
                "document_name": "Superagent",
                "score": 0.9,
                "reasoning": "Highly relevant"
            }
        ]
    }
    
    # Mock the ranking LLM call
    with patch('reag.ranking.acompletion', new_callable=AsyncMock) as mock_rank:
        mock_rank.return_value = Mock(
            choices=[
                Mock(
                    message=Mock(
                        content=json.dumps(ranked_response)
                    )
                )
            ]
        )
        
        # Create client with ranking enabled
        rag_config = RagConfig(rank_results=True)
        async with ReagClient(rag_config=rag_config) as client:
            response = await client.query("What is Superagent?", documents=SAMPLE_DOCS)
            
            assert len(response) == 1
            assert isinstance(response[0].document, Document)
            assert response[0].score == 0.9
            assert response[0].reasoning == "Highly relevant"

@pytest.mark.asyncio
async def test_batch_query_with_ranking(mock_llm):
    """Test batch query with document ranking enabled."""
    # Create sample ranked response
    ranked_response = {
        "rankings": [
            {
                "document_name": "Superagent",
                "score": 0.9,
                "reasoning": "Highly relevant"
            }
        ]
    }
    
    # Mock the ranking LLM call
    with patch('reag.ranking.acompletion', new_callable=AsyncMock) as mock_rank:
        mock_rank.return_value = Mock(
            choices=[
                Mock(
                    message=Mock(
                        content=json.dumps(ranked_response)
                    )
                )
            ]
        )
        
        # Create client with ranking enabled
        rag_config = RagConfig(rank_results=True)
        async with ReagClient(rag_config=rag_config) as client:
            queries = ["What is Superagent?", "Tell me about Superagent"]
            results = await client.query_batch(queries=queries, documents=SAMPLE_DOCS)
            
            assert len(results) == 2
            for result in results:
                assert len(result.results) == 1
                assert result.error is None
                # Verify the results were converted back to QueryResult format
                query_result = result.results[0]
                assert query_result.is_irrelevant is False  # score was 0.9, above threshold
                assert query_result.document.name == "Superagent"
                assert query_result.reasoning == "Highly relevant"

@pytest.mark.asyncio
async def test_query_with_ranking_and_storage(mock_llm):
    """Test query with document ranking and result storage enabled."""
    # Create sample ranked response
    ranked_response = {
        "rankings": [
            {
                "document_name": "Superagent",
                "score": 0.9,
                "reasoning": "Highly relevant"
            }
        ]
    }
    
    # Mock the ranking LLM call and storage
    with patch('reag.ranking.acompletion', new_callable=AsyncMock) as mock_rank, \
         patch('reag.ranking_storage.RankingStorage') as MockStorage:
        # Setup ranking mock
        mock_rank.return_value = Mock(
            choices=[
                Mock(
                    message=Mock(
                        content=json.dumps(ranked_response)
                    )
                )
            ]
        )
        
        # Setup storage mock
        mock_storage = MockStorage.return_value
        mock_storage.store_ranking_results = AsyncMock(return_value="test-result-id")
        
        # Create client with ranking and storage enabled
        rag_config = RagConfig(rank_results=True)
        async with ReagClient(
            rag_config=rag_config,
            model="test-model",
            model_kwargs={"temperature": 0.7}
        ) as client:
            # Initialize ranker with storage enabled
            client._ranker = DocumentRanker(
                model=client.model,
                model_kwargs=client.model_kwargs,
                store_results=True
            )
            # Set the mocked storage instance
            client._ranker._storage = mock_storage
            
            # Execute query
            response = await client.query(
                "What is Superagent?",
                documents=SAMPLE_DOCS,
                store_metadata={"environment": "test"}
            )
            
            # Verify ranking results
            assert len(response) == 1
            assert isinstance(response[0].document, Document)
            assert response[0].score == 0.9
            assert response[0].reasoning == "Highly relevant"
            
            # Verify storage was called with correct data
            mock_storage.store_ranking_results.assert_called_once()
            call_args = mock_storage.store_ranking_results.call_args[1]
            assert call_args["query"] == "What is Superagent?"
            assert len(call_args["ranked_docs"]) == 1
            assert call_args["metadata"]["environment"] == "test"
            assert call_args["metadata"]["model"] == "test-model"
            assert call_args["metadata"]["model_param_temperature"] == "0.7"

@pytest.mark.asyncio
async def test_batch_query_with_ranking_and_storage(mock_llm):
    """Test batch query with document ranking and result storage enabled."""
    # Create sample ranked response
    ranked_response = {
        "rankings": [
            {
                "document_name": "Superagent",
                "score": 0.9,
                "reasoning": "Highly relevant"
            }
        ]
    }
    
    # Mock the ranking LLM call and storage
    with patch('reag.ranking.acompletion', new_callable=AsyncMock) as mock_rank, \
         patch('reag.ranking_storage.RankingStorage') as MockStorage:
        # Setup ranking mock
        mock_rank.return_value = Mock(
            choices=[
                Mock(
                    message=Mock(
                        content=json.dumps(ranked_response)
                    )
                )
            ]
        )
        
        # Setup storage mock
        mock_storage = MockStorage.return_value
        mock_storage.store_ranking_results = AsyncMock(return_value="test-result-id")
        
        # Create client with ranking and storage enabled
        rag_config = RagConfig(rank_results=True)
        async with ReagClient(
            rag_config=rag_config,
            model="test-model",
            model_kwargs={"temperature": 0.7}
        ) as client:
            # Initialize ranker with storage enabled
            client._ranker = DocumentRanker(
                model=client.model,
                model_kwargs=client.model_kwargs,
                store_results=True
            )
            # Set the mocked storage instance
            client._ranker._storage = mock_storage
            
            # Execute batch query
            queries = ["What is Superagent?", "Tell me about Superagent"]
            results = await client.query_batch(
                queries=queries,
                documents=SAMPLE_DOCS,
                store_metadata={"environment": "test"}
            )
            
            # Verify batch results
            assert len(results) == 2
            for result in results:
                assert len(result.results) == 1
                assert result.error is None
                query_result = result.results[0]
                assert query_result.is_irrelevant is False
                assert query_result.document.name == "Superagent"
                assert query_result.reasoning == "Highly relevant"
            
            # Verify storage was called for each query
            assert mock_storage.store_ranking_results.call_count == 2
            for call_args in mock_storage.store_ranking_results.call_args_list:
                args = call_args[1]
                assert args["query"] in queries
                assert len(args["ranked_docs"]) == 1
                assert args["metadata"]["environment"] == "test"
                assert args["metadata"]["model"] == "test-model"
                assert args["metadata"]["model_param_temperature"] == "0.7"
