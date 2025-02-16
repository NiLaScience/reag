import pytest
from unittest.mock import Mock, patch, AsyncMock
import json

from reag.ranking import DocumentRanker, RankedDocument
from reag.schema import Document, QueryResult

# Test data
SAMPLE_DOCS = [
    Document(
        name="doc1",
        content="This is a highly relevant document",
        metadata={"category": "test"}
    ),
    Document(
        name="doc2",
        content="This is somewhat relevant",
        metadata={"category": "test"}
    ),
    Document(
        name="doc3",
        content="This is not very relevant",
        metadata={"category": "test"}
    )
]

SAMPLE_RESULTS = [
    QueryResult(
        content="Content 1",
        reasoning="Initial analysis 1",
        is_irrelevant=False,
        document=SAMPLE_DOCS[0]
    ),
    QueryResult(
        content="Content 2",
        reasoning="Initial analysis 2",
        is_irrelevant=False,
        document=SAMPLE_DOCS[1]
    ),
    QueryResult(
        content="Content 3",
        reasoning="Initial analysis 3",
        is_irrelevant=False,
        document=SAMPLE_DOCS[2]
    )
]

MOCK_RANKING_RESPONSE = {
    "rankings": [
        {
            "document_name": "doc1",
            "score": 0.9,
            "reasoning": "Highly relevant to query"
        },
        {
            "document_name": "doc2",
            "score": 0.6,
            "reasoning": "Somewhat relevant"
        },
        {
            "document_name": "doc3",
            "score": 0.3,
            "reasoning": "Not very relevant"
        }
    ]
}

@pytest.fixture
def mock_llm():
    """Mock LLM completion calls."""
    with patch('reag.ranking.acompletion', new_callable=AsyncMock) as mock_completion:
        mock_completion.return_value = Mock(
            choices=[
                Mock(
                    message=Mock(
                        content=json.dumps(MOCK_RANKING_RESPONSE)
                    )
                )
            ]
        )
        yield mock_completion

@pytest.mark.asyncio
async def test_rank_documents(mock_llm):
    """Test document ranking with multiple documents."""
    ranker = DocumentRanker()
    ranked_docs = await ranker.rank_documents("test query", SAMPLE_RESULTS)
    
    assert len(ranked_docs) == 3
    assert ranked_docs[0].document.name == "doc1"
    assert ranked_docs[0].score == 0.9
    assert ranked_docs[1].document.name == "doc2"
    assert ranked_docs[1].score == 0.6
    assert ranked_docs[2].document.name == "doc3"
    assert ranked_docs[2].score == 0.3

@pytest.mark.asyncio
async def test_empty_results():
    """Test ranking with empty results list."""
    ranker = DocumentRanker()
    ranked_docs = await ranker.rank_documents("test query", [])
    assert len(ranked_docs) == 0

@pytest.mark.asyncio
async def test_error_handling(mock_llm):
    """Test error handling in document ranking."""
    mock_llm.side_effect = Exception("Ranking failed")
    
    ranker = DocumentRanker()
    with pytest.raises(Exception, match="Document ranking failed: Ranking failed"):
        await ranker.rank_documents("test query", SAMPLE_RESULTS)

@pytest.mark.asyncio
async def test_prompt_generation():
    """Test prompt generation for ranking."""
    from reag.prompt import create_ranking_prompt
    
    # Convert sample results to prompt format
    docs_for_prompt = [
        {
            "name": r.document.name,
            "content": r.document.content,
            "reasoning": r.reasoning
        }
        for r in SAMPLE_RESULTS
    ]
    
    prompt = create_ranking_prompt("test query", docs_for_prompt)
    
    # Verify prompt contains key elements
    assert 'test query' in prompt
    assert 'Document 1' in prompt
    assert 'Initial analysis 1' in prompt
    assert 'document_name' in prompt
    assert 'score' in prompt
    assert 'reasoning' in prompt 