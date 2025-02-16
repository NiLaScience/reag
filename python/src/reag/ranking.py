from typing import List, Dict, Optional, Union
from litellm import acompletion
import json

from .schema import Document, QueryResult, RankedDocument, RankingResponse
from .prompt import RANKING_SYSTEM_PROMPT, create_ranking_prompt
from .ranking_storage import RankingStorage

class DocumentRanker:
    """Ranks documents based on LLM reasoning."""
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        model_kwargs: Optional[Dict] = None,
        store_results: bool = False,
        storage_collection: str = "ranking_results"
    ):
        self.model = model
        self.model_kwargs = model_kwargs or {}
        self.store_results = store_results
        self._storage = RankingStorage(collection_name=storage_collection) if store_results else None
    
    async def rank_documents(
        self,
        query: str,
        results: List[QueryResult],
        store_metadata: Optional[Dict[str, Union[str, int]]] = None
    ) -> List[RankedDocument]:
        """
        Rank documents based on their relevance to the query using LLM reasoning.
        
        Args:
            query: The original query string
            results: List of QueryResult objects containing documents and initial analysis
            store_metadata: Optional metadata to store with ranking results
            
        Returns:
            List of RankedDocument objects sorted by score (highest first)
        """
        try:
            # Skip ranking if no results
            if not results:
                return []
            
            # Prepare documents for ranking prompt
            docs_for_prompt = [
                {
                    "name": r.document.name,
                    "content": r.document.content,
                    "reasoning": r.reasoning
                }
                for r in results
            ]
            
            # Create ranking prompt
            prompt = create_ranking_prompt(query, docs_for_prompt)
            
            # Get LLM response
            response = await acompletion(
                model=self.model,
                messages=[
                    {"role": "system", "content": RANKING_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json"},
                **self.model_kwargs
            )
            
            # Parse response
            content = response.choices[0].message.content
            rankings_data = RankingResponse.model_validate(json.loads(content))
            
            # Create RankedDocument objects
            ranked_docs = []
            doc_map = {r.document.name: r.document for r in results}
            
            for rank in rankings_data.rankings:
                doc_name = rank["document_name"]
                if doc_name in doc_map:
                    ranked_docs.append(
                        RankedDocument(
                            document=doc_map[doc_name],
                            score=float(rank["score"]),
                            reasoning=rank["reasoning"]
                        )
                    )
            
            # Sort by score descending
            ranked_docs.sort(key=lambda x: x.score, reverse=True)
            
            # Store results if enabled
            if self.store_results and self._storage:
                # Add model info to metadata
                metadata = store_metadata or {}
                metadata.update({
                    "model": self.model,
                    **{f"model_param_{k}": str(v) for k, v in self.model_kwargs.items()}
                })
                
                await self._storage.store_ranking_results(
                    query=query,
                    ranked_docs=ranked_docs,
                    metadata=metadata
                )
            
            return ranked_docs
            
        except Exception as e:
            raise Exception(f"Document ranking failed: {str(e)}") 