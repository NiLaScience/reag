import httpx
import asyncio
import json
import re
from typing import List, Optional, TypeVar, Dict, Union, Tuple
from pydantic import BaseModel
from litellm import acompletion

from reag.prompt import REAG_SYSTEM_PROMPT
from reag.schema import ResponseSchemaMessage, Document, BatchQueryResult, QueryResult
from reag.firebase_search import FirebaseConfig, FirebaseVectorSearch
from reag.ranking import DocumentRanker, RankedDocument


class MetadataFilter(BaseModel):
    key: str
    value: Union[str, int]
    operator: Optional[str] = None


class RagConfig(BaseModel):
    """Configuration for RAG prefiltering"""
    enabled: bool = False
    firebase_config: Optional[FirebaseConfig] = None
    top_k: int = 5
    min_similarity_score: Optional[float] = None
    rank_results: bool = False  # Whether to rank results using LLM


T = TypeVar("T")


class QueryResult(BaseModel):
    content: str
    reasoning: str
    is_irrelevant: bool
    document: Document


DEFAULT_BATCH_SIZE = 20


class ReagClient:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        system: str = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        schema: Optional[BaseModel] = None,
        model_kwargs: Optional[Dict] = None,
        rag_config: Optional[RagConfig] = None,
    ):
        self.model = model
        self.system = system or REAG_SYSTEM_PROMPT
        self.batch_size = batch_size
        self.schema = schema or ResponseSchemaMessage
        self.model_kwargs = model_kwargs or {}
        self.rag_config = rag_config or RagConfig()
        self._http_client = None
        self._vector_search = None
        self._ranker = None
        
        # Initialize vector search if RAG is enabled
        if self.rag_config.enabled and self.rag_config.firebase_config:
            self._vector_search = FirebaseVectorSearch(self.rag_config.firebase_config)
            
        # Initialize ranker if ranking is enabled
        if self.rag_config.rank_results:
            self._ranker = DocumentRanker(model=self.model, model_kwargs=self.model_kwargs)

    async def __aenter__(self):
        self._http_client = httpx.AsyncClient()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._http_client:
            await self._http_client.aclose()

    def _filter_documents_by_metadata(
        self, documents: List[Document], filters: Optional[List[MetadataFilter]] = None
    ) -> List[Document]:
        if not filters:
            return documents

        filtered_docs = []
        for doc in documents:
            matches_all_filters = True

            for filter_item in filters:
                metadata_value = (
                    doc.metadata.get(filter_item.key) if doc.metadata else None
                )
                if metadata_value is None:
                    matches_all_filters = False
                    break

                if isinstance(metadata_value, str) and isinstance(
                    filter_item.value, str
                ):
                    if filter_item.operator == "contains":
                        if not filter_item.value in metadata_value:
                            matches_all_filters = False
                            break
                    elif filter_item.operator == "startsWith":
                        if not metadata_value.startswith(filter_item.value):
                            matches_all_filters = False
                            break
                    elif filter_item.operator == "endsWith":
                        if not metadata_value.endswith(filter_item.value):
                            matches_all_filters = False
                            break
                    elif filter_item.operator == "regex":
                        import re

                        if not re.match(filter_item.value, metadata_value):
                            matches_all_filters = False
                            break

                if filter_item.operator == "equals":
                    if metadata_value != filter_item.value:
                        matches_all_filters = False
                        break
                elif filter_item.operator == "notEquals":
                    if metadata_value == filter_item.value:
                        matches_all_filters = False
                        break
                elif filter_item.operator == "greaterThan":
                    if not metadata_value > filter_item.value:
                        matches_all_filters = False
                        break
                elif filter_item.operator == "lessThan":
                    if not metadata_value < filter_item.value:
                        matches_all_filters = False
                        break
                elif filter_item.operator == "greaterThanOrEqual":
                    if not metadata_value >= filter_item.value:
                        matches_all_filters = False
                        break
                elif filter_item.operator == "lessThanOrEqual":
                    if not metadata_value <= filter_item.value:
                        matches_all_filters = False
                        break

            if matches_all_filters:
                filtered_docs.append(doc)

        return filtered_docs

    def _extract_think_content(self, text: str) -> tuple[str, str, bool]:
        """Extract content from think tags and parse the bulleted response format."""
        # Extract think content
        think_match = re.search(r'<think>(.*?)</think>', text, flags=re.DOTALL)
        reasoning = think_match.group(1).strip() if think_match else ""
        
        # Remove think tags and get remaining text
        remaining_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        
        # Initialize default values
        content = ""
        is_irrelevant = True
        
        # Extract is_irrelevant value
        irrelevant_match = re.search(r'\*\*isIrrelevant:\*\*\s*(true|false)', remaining_text, re.IGNORECASE)
        if irrelevant_match:
            is_irrelevant = irrelevant_match.group(1).lower() == 'true'
        
        # Extract content value
        content_match = re.search(r'\*\*Answer:\*\*\s*(.*?)(?:\n|$)', remaining_text, re.DOTALL)
        if content_match:
            content = content_match.group(1).strip()
        
        return content, reasoning, is_irrelevant

    async def _prefilter_documents(
        self,
        documents: List[Document],
        query: Optional[str] = None,
        query_doc_id: Optional[str] = None,
        query_vector: Optional[List[float]] = None,
        metadata_filters: Optional[Dict[str, Union[str, int]]] = None
    ) -> List[Document]:
        """Apply RAG prefiltering to documents if enabled."""
        if not self.rag_config.enabled or not self._vector_search:
            return documents
            
        try:
            # Get nearest documents from vector search
            nearest_docs = await self._vector_search.find_nearest_documents(
                query=query,
                query_doc_id=query_doc_id,
                query_vector=query_vector,
                limit=self.rag_config.top_k,
                metadata_filters=metadata_filters
            )
            
            # If we got results, use them; otherwise fall back to original documents
            return nearest_docs if nearest_docs else documents
            
        except Exception as e:
            print(f"RAG prefiltering failed: {str(e)}. Falling back to original documents.")
            return documents

    async def query(
        self,
        prompt: str,
        documents: List[Document],
        options: Optional[Dict] = None,
        query_doc_id: Optional[str] = None,
        query_vector: Optional[List[float]] = None,
        store_metadata: Optional[Dict[str, Union[str, int]]] = None
    ) -> List[Union[QueryResult, RankedDocument]]:
        try:
            # Get initial results
            results = await self._query_documents(
                prompt=prompt,
                documents=documents,
                options=options,
                query_doc_id=query_doc_id,
                query_vector=query_vector
            )
            
            # Rank results if enabled and we have results
            if self.rag_config.rank_results and results and self._ranker:
                ranked_results = await self._ranker.rank_documents(
                    prompt,
                    results,
                    store_metadata=store_metadata
                )
                return ranked_results
                
            return results

        except Exception as e:
            raise Exception(f"Query failed: {str(e)}")

    async def _query_documents(
        self,
        prompt: str,
        documents: List[Document],
        options: Optional[Dict] = None,
        query_doc_id: Optional[str] = None,
        query_vector: Optional[List[float]] = None,
    ) -> List[QueryResult]:
        """Internal method to query documents without ranking."""
        # Extract metadata filters
        filters = None
        if options and "filter" in options:
            raw_filters = options["filter"]
            if isinstance(raw_filters, list):
                filters = [
                    MetadataFilter(**f) if isinstance(f, dict) else f
                    for f in raw_filters
                ]
            elif isinstance(raw_filters, dict):
                filters = [MetadataFilter(**raw_filters)]

        # Convert metadata filters to dict format for vector search
        metadata_filters = {
            f.key: f.value for f in filters
        } if filters else None

        # Apply RAG prefiltering if enabled
        if self.rag_config.enabled:
            documents = await self._prefilter_documents(
                documents=documents,
                query=prompt if not (query_doc_id or query_vector) else None,
                query_doc_id=query_doc_id,
                query_vector=query_vector,
                metadata_filters=metadata_filters
            )

        # Apply traditional metadata filtering
        filtered_documents = self._filter_documents_by_metadata(documents, filters)

        def format_doc(doc: Document) -> str:
            return f"Name: {doc.name}\nMetadata: {doc.metadata}\nContent: {doc.content}"

        batch_size = self.batch_size
        batches = [
            filtered_documents[i : i + batch_size]
            for i in range(0, len(filtered_documents), batch_size)
        ]

        results = []
        for batch in batches:
            tasks = []
            # Create tasks for parallel processing within the batch
            for document in batch:
                system = f"{self.system}\n\n# Available source\n\n{format_doc(document)}"
                tasks.append(
                    acompletion(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": prompt},
                        ],
                        response_format=self.schema,
                        **self.model_kwargs,
                    )
                )

            # Process all documents in the batch concurrently
            batch_responses = await asyncio.gather(*tasks)

            # Process the responses
            for document, response in zip(batch, batch_responses):
                message_content = response.choices[0].message.content

                try:
                    if self.model.startswith("ollama/"):
                        content, reasoning, is_irrelevant = self._extract_think_content(message_content)
                        results.append(
                            QueryResult(
                                content=content,
                                reasoning=reasoning,
                                is_irrelevant=is_irrelevant,
                                document=document,
                            )
                        )
                    else:
                        # Ensure it's parsed as a dict
                        data = (
                            json.loads(message_content)
                            if isinstance(message_content, str)
                            else message_content
                        )

                        if data["source"].get("is_irrelevant", True):
                            continue

                        results.append(
                            QueryResult(
                                content=data["source"].get("content", ""),
                                reasoning=data["source"].get("reasoning", ""),
                                is_irrelevant=data["source"].get("is_irrelevant", False),
                                document=document,
                            )
                        )
                except json.JSONDecodeError:
                    print("Error: Could not parse response:", message_content)
                    continue

        return results

    async def query_batch(
        self,
        queries: List[str],
        documents: List[Document],
        options: Optional[Dict] = None,
        max_concurrent: int = 5,
        query_doc_id: Optional[str] = None,
        query_vector: Optional[List[float]] = None,
        store_metadata: Optional[Dict[str, Union[str, int]]] = None
    ) -> List[BatchQueryResult]:
        """
        Process multiple queries in parallel against the same document set.
        
        Args:
            queries: List of query strings to process
            documents: List of documents to search through
            options: Optional filtering and configuration options
            max_concurrent: Maximum number of concurrent queries to process
            query_doc_id: Optional document ID to use for vector search
            query_vector: Optional vector to use for similarity search
            store_metadata: Optional metadata to store with ranking results
            
        Returns:
            List of BatchQueryResult objects containing results for each query
        """
        try:
            # Create semaphore for concurrency control
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def process_query(query: str) -> BatchQueryResult:
                try:
                    async with semaphore:
                        results = await self.query(
                            prompt=query,
                            documents=documents,
                            options=options,
                            query_doc_id=query_doc_id,
                            query_vector=query_vector,
                            store_metadata=store_metadata
                        )
                        
                        # Convert results to appropriate format
                        if results and isinstance(results[0], RankedDocument):
                            # Convert ranked documents back to QueryResults for batch response
                            query_results = [
                                QueryResult(
                                    content=r.document.content,
                                    reasoning=r.reasoning,
                                    is_irrelevant=r.score < 0.5,  # Use score threshold
                                    document=r.document
                                )
                                for r in results
                            ]
                        else:
                            query_results = results
                        
                        # Convert QueryResult objects to dictionaries
                        results_dict = [result.model_dump() for result in query_results]
                        
                        return BatchQueryResult(
                            query=query,
                            results=results_dict,
                            error=None
                        )
                except Exception as e:
                    return BatchQueryResult(
                        query=query,
                        results=[],
                        error=str(e)
                    )
            
            # Create tasks for all queries
            tasks = [process_query(query) for query in queries]
            
            # Execute all tasks concurrently and gather results
            results = await asyncio.gather(*tasks)
            
            return results
            
        except Exception as e:
            raise Exception(f"Batch query processing failed: {str(e)}")
