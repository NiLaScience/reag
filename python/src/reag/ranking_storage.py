from typing import List, Dict, Optional, Union
from datetime import datetime
from google.cloud import firestore
from pydantic import BaseModel

from .schema import RankedDocument, RankingResultStorage


class RankingStorage:
    """Handles storage of ranking results in Firebase."""
    
    def __init__(
        self,
        collection_name: str = "ranking_results",
        db: Optional[firestore.Client] = None
    ):
        self.collection_name = collection_name
        self.db = db or firestore.Client()
        self.collection = self.db.collection(collection_name)
    
    async def store_ranking_results(
        self,
        query: str,
        ranked_docs: List[RankedDocument],
        metadata: Optional[Dict[str, Union[str, int]]] = None
    ) -> str:
        """
        Store ranking results in Firebase.
        
        Args:
            query: The original query string
            ranked_docs: List of ranked documents
            metadata: Optional metadata about the ranking (e.g., model used, parameters)
            
        Returns:
            ID of the stored document
        """
        try:
            # Convert ranked documents to storable format
            results = [
                {
                    "document_name": doc.document.name,
                    "document_content": doc.document.content,
                    "document_metadata": doc.document.metadata,
                    "score": doc.score,
                    "reasoning": doc.reasoning
                }
                for doc in ranked_docs
            ]
            
            # Create storage object
            storage_obj = RankingResultStorage(
                query=query,
                timestamp=datetime.utcnow().isoformat(),
                results=results,
                metadata=metadata
            )
            
            # Store in Firebase
            doc_ref = self.collection.document()
            doc_ref.set(storage_obj.model_dump())
            
            return doc_ref.id
            
        except Exception as e:
            raise Exception(f"Failed to store ranking results: {str(e)}")
    
    async def get_ranking_results(
        self,
        limit: int = 10,
        metadata_filters: Optional[Dict[str, Union[str, int]]] = None
    ) -> List[RankingResultStorage]:
        """
        Retrieve stored ranking results.
        
        Args:
            limit: Maximum number of results to return
            metadata_filters: Optional filters to apply on metadata fields
            
        Returns:
            List of RankingResultStorage objects
        """
        try:
            # Start with base query
            query = self.collection.order_by("timestamp", direction=firestore.Query.DESCENDING)
            
            # Apply metadata filters if provided
            if metadata_filters:
                for key, value in metadata_filters.items():
                    query = query.where(f"metadata.{key}", "==", value)
            
            # Execute query with limit
            docs = query.limit(limit).get()
            
            # Convert to RankingResultStorage objects
            return [
                RankingResultStorage.model_validate(doc.to_dict())
                for doc in docs
            ]
            
        except Exception as e:
            raise Exception(f"Failed to retrieve ranking results: {str(e)}")
    
    async def delete_ranking_result(self, result_id: str) -> None:
        """
        Delete a specific ranking result.
        
        Args:
            result_id: ID of the result document to delete
        """
        try:
            self.collection.document(result_id).delete()
        except Exception as e:
            raise Exception(f"Failed to delete ranking result: {str(e)}") 