from typing import Optional, Dict, Union, Literal, List, ForwardRef
from pydantic import BaseModel


class Document(BaseModel):
    """A document with content and optional metadata."""
    name: str
    content: str
    metadata: Optional[Dict[str, Union[str, int]]] = None


class ResponseSchema(BaseModel):
    content: str
    reasoning: str
    is_irrelevant: bool


class ResponseSchemaMessage(BaseModel):
    """Schema for LLM responses."""
    source: Dict[str, Union[str, bool]]


class MetadataFilter(BaseModel):
    """Filter for document metadata."""
    key: str
    value: Union[str, int]
    operator: Optional[Literal["equals", "greaterThan", "lessThan", "greaterThanOrEqual", "lessThanOrEqual"]] = "equals"


class QueryResult(BaseModel):
    """Result of a single document query."""
    content: str
    reasoning: str
    is_irrelevant: bool
    document: Document


class RankingResponse(BaseModel):
    """Schema for document ranking responses."""
    rankings: List[Dict[str, Union[str, float]]]


class RankingResultStorage(BaseModel):
    """Schema for storing ranking results in Firebase."""
    query: str
    timestamp: str
    results: List[Dict[str, Union[str, float, Dict]]]
    metadata: Optional[Dict[str, Union[str, int]]] = None


class RankedDocument(BaseModel):
    """A document with its ranking score and reasoning."""
    document: Document
    score: float  # 0-1 score indicating relevance
    reasoning: str  # Explanation for the score


class BatchQueryResult(BaseModel):
    """Result of a single query in a batch operation."""
    query: str
    results: List[QueryResult]
    error: Optional[str] = None

# Rebuild models to resolve forward references
BatchQueryResult.model_rebuild()
