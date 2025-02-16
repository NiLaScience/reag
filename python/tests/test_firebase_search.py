import pytest
from unittest.mock import Mock, patch, AsyncMock
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
from google.cloud.firestore_v1.vector import Vector

from reag.firebase_search import FirebaseConfig, FirebaseVectorSearch
from reag.schema import Document

# Test data
SAMPLE_EMBEDDING = [0.1, 0.2, 0.3]
SAMPLE_DOC_DATA = {
    "name": "test_doc",
    "content": "test content",
    "metadata": {"category": "test"},
    "embedding": SAMPLE_EMBEDDING
}

@pytest.fixture
def mock_firestore():
    """Create a mock Firestore client."""
    # Create mock document
    mock_doc = Mock()
    mock_doc.exists = True
    mock_doc.get.side_effect = lambda x, default=None: SAMPLE_DOC_DATA.get(x, default)
    
    # Create mock collection query
    mock_query = Mock()
    mock_query.where.return_value = mock_query
    mock_query.get.return_value = [mock_doc]
    
    # Create mock collection
    mock_collection = Mock()
    mock_collection.document.return_value.get.return_value = mock_doc
    mock_collection.find_nearest.return_value = mock_query
    
    # Make collection behave like a query
    mock_collection.where.return_value = mock_query
    mock_collection.get.return_value = [mock_doc]
    
    # Create mock db
    mock_db = Mock()
    mock_db.collection.return_value = mock_collection
    
    return mock_db

@pytest.fixture
def firebase_search(mock_firestore):
    """Create a FirebaseVectorSearch instance with mocked Firestore."""
    config = FirebaseConfig(collection_name="test_collection")
    return FirebaseVectorSearch(config=config, db=mock_firestore)

@pytest.mark.asyncio
async def test_get_embedding():
    """Test embedding generation."""
    config = FirebaseConfig(collection_name="test_collection")
    search = FirebaseVectorSearch(config=config)
    
    with patch('reag.firebase_search.aembedding', new_callable=AsyncMock) as mock_embed:
        mock_embed.return_value.data = [Mock(embedding=SAMPLE_EMBEDDING)]
        
        result = await search.get_embedding("test query")
        assert result == SAMPLE_EMBEDDING

@pytest.mark.asyncio
async def test_get_document_embedding(firebase_search):
    """Test retrieving embedding from existing document."""
    result = await firebase_search.get_document_embedding("test_id")
    assert result == SAMPLE_EMBEDDING
    
    # Test non-existent document
    firebase_search.collection.document.return_value.get.return_value.exists = False
    result = await firebase_search.get_document_embedding("nonexistent")
    assert result is None

@pytest.mark.asyncio
async def test_find_nearest_documents(firebase_search):
    """Test vector similarity search with different query types."""
    # Test with text query
    with patch.object(firebase_search, 'get_embedding', new_callable=AsyncMock) as mock_embed:
        mock_embed.return_value = SAMPLE_EMBEDDING
        docs = await firebase_search.find_nearest_documents(query="test query")
        assert len(docs) == 1
        assert docs[0].name == "test_doc"
        assert docs[0].content == "test content"
        assert docs[0].metadata == {"category": "test"}

@pytest.mark.asyncio
async def test_metadata_filtering(firebase_search):
    """Test metadata filtering in vector search."""
    docs = await firebase_search.find_nearest_documents(
        query_vector=SAMPLE_EMBEDDING,
        metadata_filters={"category": "test"}
    )
    assert len(docs) == 1
    assert docs[0].name == "test_doc"
    assert docs[0].metadata == {"category": "test"}

@pytest.mark.asyncio
async def test_error_handling(firebase_search):
    """Test error handling scenarios."""
    # Test missing query parameters
    with pytest.raises(ValueError, match="Must provide either query, query_doc_id, or query_vector"):
        await firebase_search.find_nearest_documents()
    
    # Test embedding generation failure
    with patch.object(firebase_search, 'get_embedding', side_effect=Exception("Embedding failed")):
        with pytest.raises(Exception, match="Embedding failed"):
            await firebase_search.find_nearest_documents(query="test")

@pytest.mark.asyncio
async def test_load_documents(firebase_search):
    """Test loading documents from a collection."""
    docs = await firebase_search.load_documents()
    assert len(docs) == 1
    assert docs[0].name == "test_doc"
    assert docs[0].content == "test content"
    assert docs[0].metadata == {"category": "test"}
    
    # Test with metadata filters
    docs = await firebase_search.load_documents(metadata_filters={"category": "test"})
    assert len(docs) == 1
    assert docs[0].metadata["category"] == "test"

@pytest.mark.asyncio
async def test_query_collection_to_collection(firebase_search):
    """Test querying between collections."""
    # Mock load_documents to return different docs for query and target collections
    query_docs = [
        Document(
            name="query_doc",
            content="query content",
            metadata={"type": "query"}
        )
    ]
    
    target_docs = [
        Document(
            name="target_doc",
            content="target content",
            metadata={"type": "target"}
        )
    ]
    
    with patch.object(firebase_search, 'load_documents', new_callable=AsyncMock) as mock_load:
        mock_load.side_effect = [query_docs, target_docs]
        
        with patch.object(firebase_search, 'get_embedding', new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = [0.1, 0.2, 0.3]
            
            with patch.object(firebase_search, 'find_nearest_documents', new_callable=AsyncMock) as mock_find:
                mock_find.return_value = target_docs
                
                results = await firebase_search.query_collection_to_collection(
                    query_collection="queries",
                    doc_collection="targets",
                    query_metadata_filters={"type": "query"},
                    doc_metadata_filters={"type": "target"}
                )
                
                assert len(results) == 1
                assert "query_doc" in results
                assert len(results["query_doc"]) == 1
                assert results["query_doc"][0].name == "target_doc"
                
                # Verify correct method calls
                mock_load.assert_any_call(collection_name="queries", metadata_filters={"type": "query"})
                mock_load.assert_any_call(collection_name="targets", metadata_filters={"type": "target"})
                mock_embed.assert_called_once_with("query content")
                mock_find.assert_called_once_with(
                    query_vector=[0.1, 0.2, 0.3],
                    limit=5,
                    metadata_filters={"type": "target"}
                ) 