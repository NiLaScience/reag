import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
import json

from reag.ranking_storage import RankingStorage
from reag.schema import Document, RankedDocument, RankingResultStorage

# Test data
SAMPLE_DOCS = [
    Document(
        name="doc1",
        content="Test content 1",
        metadata={"category": "test"}
    ),
    Document(
        name="doc2",
        content="Test content 2",
        metadata={"category": "other"}
    )
]

SAMPLE_RANKED_DOCS = [
    RankedDocument(
        document=SAMPLE_DOCS[0],
        score=0.9,
        reasoning="Highly relevant"
    ),
    RankedDocument(
        document=SAMPLE_DOCS[1],
        score=0.5,
        reasoning="Somewhat relevant"
    )
]

@pytest.fixture
def mock_firestore():
    """Create a mock Firestore client."""
    # Create mock document
    mock_doc = Mock()
    mock_doc.to_dict.return_value = {
        "query": "test query",
        "timestamp": datetime.utcnow().isoformat(),
        "results": [
            {
                "document_name": "doc1",
                "document_content": "Test content 1",
                "document_metadata": {"category": "test"},
                "score": 0.9,
                "reasoning": "Highly relevant"
            }
        ],
        "metadata": {"model": "test-model"}
    }
    
    # Create mock collection
    mock_collection = Mock()
    mock_collection.document.return_value = mock_doc
    mock_collection.order_by.return_value = mock_collection
    mock_collection.where.return_value = mock_collection
    mock_collection.limit.return_value = mock_collection
    mock_collection.get.return_value = [mock_doc]
    
    # Create mock db
    mock_db = Mock()
    mock_db.collection.return_value = mock_collection
    
    return mock_db

@pytest.fixture
def storage(mock_firestore):
    """Create a RankingStorage instance with mocked Firestore."""
    return RankingStorage(collection_name="test_rankings", db=mock_firestore)

@pytest.mark.asyncio
async def test_store_ranking_results(storage):
    """Test storing ranking results in Firebase."""
    # Test metadata
    metadata = {
        "model": "test-model",
        "environment": "test"
    }
    
    # Store results
    result_id = await storage.store_ranking_results(
        query="test query",
        ranked_docs=SAMPLE_RANKED_DOCS,
        metadata=metadata
    )
    
    assert result_id is not None
    
    # Verify document was created with correct data
    doc_ref = storage.collection.document()
    doc_ref.set.assert_called_once()
    
    # Verify stored data structure
    call_args = doc_ref.set.call_args[0][0]
    assert call_args["query"] == "test query"
    assert "timestamp" in call_args
    assert len(call_args["results"]) == 2
    assert call_args["metadata"] == metadata
    
    # Verify document data
    first_result = call_args["results"][0]
    assert first_result["document_name"] == "doc1"
    assert first_result["score"] == 0.9
    assert first_result["reasoning"] == "Highly relevant"

@pytest.mark.asyncio
async def test_get_ranking_results(storage):
    """Test retrieving ranking results."""
    results = await storage.get_ranking_results(
        limit=10,
        metadata_filters={"model": "test-model"}
    )
    
    assert len(results) == 1
    assert isinstance(results[0], RankingResultStorage)
    assert results[0].query == "test query"
    assert len(results[0].results) == 1
    assert results[0].metadata == {"model": "test-model"}
    
    # Verify query construction
    storage.collection.order_by.assert_called_once_with(
        "timestamp", 
        direction="DESCENDING"
    )
    storage.collection.limit.assert_called_once_with(10)

@pytest.mark.asyncio
async def test_get_ranking_results_with_filters(storage):
    """Test retrieving results with metadata filters."""
    filters = {
        "model": "test-model",
        "environment": "test"
    }
    
    results = await storage.get_ranking_results(
        metadata_filters=filters
    )
    
    # Verify filter application
    for key, value in filters.items():
        storage.collection.where.assert_any_call(
            f"metadata.{key}",
            "==",
            value
        )

@pytest.mark.asyncio
async def test_delete_ranking_result(storage):
    """Test deleting a ranking result."""
    result_id = "test-id"
    await storage.delete_ranking_result(result_id)
    
    # Verify deletion
    storage.collection.document.assert_called_once_with(result_id)
    storage.collection.document.return_value.delete.assert_called_once()

@pytest.mark.asyncio
async def test_error_handling(storage):
    """Test error handling in storage operations."""
    # Test store error
    storage.collection.document.return_value.set.side_effect = Exception("Store failed")
    with pytest.raises(Exception, match="Failed to store ranking results: Store failed"):
        await storage.store_ranking_results("query", SAMPLE_RANKED_DOCS)
    
    # Test retrieve error
    storage.collection.get.side_effect = Exception("Retrieve failed")
    with pytest.raises(Exception, match="Failed to retrieve ranking results: Retrieve failed"):
        await storage.get_ranking_results()
    
    # Test delete error
    storage.collection.document.return_value.delete.side_effect = Exception("Delete failed")
    with pytest.raises(Exception, match="Failed to delete ranking result: Delete failed"):
        await storage.delete_ranking_result("test-id") 