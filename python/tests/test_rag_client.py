import pytest
from unittest.mock import Mock, patch, AsyncMock
from typing import List

from reag.client import ReagClient, RagConfig, Document, QueryResult
from reag.firebase_search import FirebaseConfig

# Test data
SAMPLE_DOCS = [
    Document(
        name="doc1",
        content="test content 1",
        metadata={"category": "test"}
    ),
    Document(
        name="doc2",
        content="test content 2",
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

@pytest.fixture
def rag_client():
    """Create a ReagClient with RAG enabled."""
    config = RagConfig(
        enabled=True,
        firebase_config=FirebaseConfig(collection_name="test_collection"),
        top_k=2
    )
    return ReagClient(model="test-model", rag_config=config)

@pytest.mark.asyncio
async def test_rag_prefiltering(rag_client):
    """Test RAG prefiltering with different query types."""
    # Mock vector search results
    mock_nearest_docs = [SAMPLE_DOCS[0]]
    
    with patch.object(rag_client._vector_search, 'find_nearest_documents', new_callable=AsyncMock) as mock_search:
        mock_search.return_value = mock_nearest_docs
        
        # Mock LLM completion
        with patch('reag.client.acompletion', new_callable=AsyncMock) as mock_completion:
            mock_completion.return_value = MOCK_RESPONSE
            
            # Test with text query
            results = await rag_client.query(
                prompt="test query",
                documents=SAMPLE_DOCS
            )
            assert len(results) == 1
            assert results[0].document == SAMPLE_DOCS[0]
            assert not results[0].is_irrelevant

@pytest.mark.asyncio
async def test_rag_with_metadata_filtering(rag_client):
    """Test RAG prefiltering with metadata filters."""
    mock_nearest_docs = [SAMPLE_DOCS[0]]
    
    with patch.object(rag_client._vector_search, 'find_nearest_documents', new_callable=AsyncMock) as mock_search:
        mock_search.return_value = mock_nearest_docs
        
        with patch('reag.client.acompletion', new_callable=AsyncMock) as mock_completion:
            mock_completion.return_value = MOCK_RESPONSE
            
            results = await rag_client.query(
                prompt="test query",
                documents=SAMPLE_DOCS,
                options={"filter": {"key": "category", "value": "test"}}
            )
            assert len(results) == 1
            assert results[0].document.metadata["category"] == "test"

@pytest.mark.asyncio
async def test_full_query_with_rag(rag_client):
    """Test full query process with RAG enabled."""
    # Mock vector search
    with patch.object(rag_client._vector_search, 'find_nearest_documents', new_callable=AsyncMock) as mock_search:
        mock_search.return_value = [SAMPLE_DOCS[0]]

        # Mock LLM completion
        with patch('reag.client.acompletion', new_callable=AsyncMock) as mock_completion:
            mock_completion.return_value = MOCK_RESPONSE

            results = await rag_client.query(
                prompt="test query",
                documents=SAMPLE_DOCS,
                options={"filter": {"key": "category", "value": "test"}}
            )

            assert len(results) == 1
            assert results[0].document == SAMPLE_DOCS[0]
            assert not results[0].is_irrelevant

@pytest.mark.asyncio
async def test_rag_fallback_behavior(rag_client):
    """Test fallback to original documents when RAG fails."""
    with patch.object(rag_client._vector_search, 'find_nearest_documents', side_effect=Exception("Search failed")):
        filtered_docs = await rag_client._prefilter_documents(
            documents=SAMPLE_DOCS,
            query="test"
        )
        
        # Should fall back to original documents
        assert len(filtered_docs) == 2
        assert filtered_docs == SAMPLE_DOCS

@pytest.mark.asyncio
async def test_rag_disabled(rag_client):
    """Test behavior when RAG is disabled."""
    rag_client.rag_config.enabled = False
    
    with patch.object(rag_client._vector_search, 'find_nearest_documents', new_callable=AsyncMock) as mock_search:
        filtered_docs = await rag_client._prefilter_documents(
            documents=SAMPLE_DOCS,
            query="test"
        )
        
        # Should return original documents without calling vector search
        assert len(filtered_docs) == 2
        assert filtered_docs == SAMPLE_DOCS
        mock_search.assert_not_called() 