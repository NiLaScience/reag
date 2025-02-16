from typing import List, Optional, Dict, Union
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
from google.cloud.firestore_v1.vector import Vector
from google.cloud import firestore
from pydantic import BaseModel
from litellm import aembedding

from .schema import Document


class FirebaseConfig(BaseModel):
    collection_name: str
    embedding_field: str = "embedding"
    content_field: str = "content"
    name_field: str = "name"
    metadata_field: str = "metadata"
    distance_measure: str = "EUCLIDEAN"
    embedding_model: str = "text-embedding-3-small"  # OpenAI's latest embedding model


class FirebaseVectorSearch:
    def __init__(
        self,
        config: FirebaseConfig,
        db: Optional[firestore.Client] = None,
    ):
        self.config = config
        self.db = db or firestore.Client()
        self.collection = self.db.collection(config.collection_name)
    
    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding vector for a text using litellm."""
        try:
            response = await aembedding(
                model=self.config.embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            raise Exception(f"Failed to generate embedding: {str(e)}")
    
    async def get_document_embedding(self, doc_id: str) -> Optional[List[float]]:
        """Get embedding from an existing Firebase document."""
        try:
            doc = self.collection.document(doc_id).get()
            if not doc.exists:
                return None
            embedding = doc.get(self.config.embedding_field)
            return embedding if embedding else None
        except Exception as e:
            raise Exception(f"Failed to get document embedding: {str(e)}")
    
    def _convert_to_document(self, doc_snapshot) -> Document:
        """Convert Firestore document to ReAG Document."""
        return Document(
            name=doc_snapshot.get(self.config.name_field),
            content=doc_snapshot.get(self.config.content_field),
            metadata=doc_snapshot.get(self.config.metadata_field, {})
        )

    async def find_nearest_documents(
        self,
        query: Optional[str] = None,
        query_doc_id: Optional[str] = None,
        query_vector: Optional[List[float]] = None,
        limit: int = 5,
        metadata_filters: Optional[Dict[str, Union[str, int]]] = None
    ) -> List[Document]:
        """
        Find nearest documents using vector similarity search.
        
        Args:
            query: The text query to search for (will be embedded)
            query_doc_id: ID of an existing Firebase document to use its embedding
            query_vector: Direct vector to use for similarity search
            limit: Maximum number of documents to return
            metadata_filters: Optional metadata filters to apply
            
        Returns:
            List of Document objects sorted by relevance
        """
        # Determine which vector to use for search
        if query_vector is not None:
            search_vector = query_vector
        elif query_doc_id is not None:
            search_vector = await self.get_document_embedding(query_doc_id)
            if not search_vector:
                raise Exception(f"No embedding found for document {query_doc_id}")
        elif query is not None:
            search_vector = await self.get_embedding(query)
        else:
            raise ValueError("Must provide either query, query_doc_id, or query_vector")
        
        # Create vector query
        vector_query = self.collection.find_nearest(
            vector_field=self.config.embedding_field,
            query_vector=Vector(search_vector),
            distance_measure=getattr(DistanceMeasure, self.config.distance_measure),
            limit=limit,
        )
        
        # Apply metadata filters if provided
        if metadata_filters:
            for key, value in metadata_filters.items():
                vector_query = vector_query.where(f"{self.config.metadata_field}.{key}", "==", value)
        
        # Execute query and convert results
        results = vector_query.get()
        return [self._convert_to_document(doc) for doc in results]

    async def load_documents(
        self,
        collection_name: Optional[str] = None,
        metadata_filters: Optional[Dict[str, Union[str, int]]] = None
    ) -> List[Document]:
        """
        Load documents from a Firestore collection.
        
        Args:
            collection_name: Optional name of collection to load from (defaults to configured collection)
            metadata_filters: Optional metadata filters to apply
            
        Returns:
            List of Document objects
        """
        try:
            # Use specified collection or default to configured one
            collection = self.db.collection(collection_name) if collection_name else self.collection
            
            # Start with base query
            query = collection
            
            # Apply metadata filters if provided
            if metadata_filters:
                for key, value in metadata_filters.items():
                    query = query.where(f"{self.config.metadata_field}.{key}", "==", value)
            
            # Execute query
            docs = query.get()
            
            # Convert to Document objects
            return [
                Document(
                    name=doc.get(self.config.name_field),
                    content=doc.get(self.config.content_field),
                    metadata=doc.get(self.config.metadata_field, {})
                )
                for doc in docs
            ]
            
        except Exception as e:
            raise Exception(f"Failed to load documents: {str(e)}")

    async def query_collection_to_collection(
        self,
        query_collection: str,
        doc_collection: str,
        query_metadata_filters: Optional[Dict[str, Union[str, int]]] = None,
        doc_metadata_filters: Optional[Dict[str, Union[str, int]]] = None,
        limit: int = 5,
    ) -> Dict[str, List[Document]]:
        """
        Query documents from one collection against another collection using vector similarity.
        
        Args:
            query_collection: Name of collection containing query documents
            doc_collection: Name of collection containing documents to search
            query_metadata_filters: Optional metadata filters for query documents
            doc_metadata_filters: Optional metadata filters for target documents
            limit: Maximum number of similar documents to return per query
            
        Returns:
            Dictionary mapping query document names to lists of similar documents
        """
        try:
            # Load query documents
            query_docs = await self.load_documents(
                collection_name=query_collection,
                metadata_filters=query_metadata_filters
            )
            
            # Load target documents
            target_docs = await self.load_documents(
                collection_name=doc_collection,
                metadata_filters=doc_metadata_filters
            )
            
            results = {}
            
            # Process each query document
            for query_doc in query_docs:
                # Get embedding for query document
                query_embedding = await self.get_embedding(query_doc.content)
                
                # Find similar documents
                similar_docs = await self.find_nearest_documents(
                    query_vector=query_embedding,
                    limit=limit,
                    metadata_filters=doc_metadata_filters
                )
                
                results[query_doc.name] = similar_docs
            
            return results
            
        except Exception as e:
            raise Exception(f"Collection-to-collection query failed: {str(e)}") 