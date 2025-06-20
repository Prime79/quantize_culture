"""Qdrant vector database service wrapper."""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, 
    FieldCondition, Range, MatchValue, CollectionStatus
)
from typing import List, Dict, Optional, Any
import uuid
from ..utils.config import config
from ..data.models import Sentence

class QdrantService:
    """Service for interacting with Qdrant vector database."""
    
    def __init__(self, host: Optional[str] = None, port: Optional[int] = None):
        """Initialize Qdrant service."""
        self.host = host or config.qdrant_host
        self.port = port or config.qdrant_port
        self.client = QdrantClient(host=self.host, port=self.port)
        self.default_collection = config.default_collection
    
    def create_collection(self, collection_name: str, vector_size: int = 1536) -> bool:
        """
        Create a new collection in Qdrant.
        
        Args:
            collection_name: Name of the collection
            vector_size: Dimension of vectors (default: 1536 for OpenAI embeddings)
            
        Returns:
            True if successful
        """
        try:
            self.client.recreate_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            return True
        except Exception as e:
            raise Exception(f"Failed to create collection '{collection_name}': {str(e)}")
    
    def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists."""
        try:
            collections = self.client.get_collections()
            return any(c.name == collection_name for c in collections.collections)
        except:
            return False
    
    def store_embeddings(self, sentences: List[Sentence], collection_name: Optional[str] = None) -> int:
        """
        Store sentence embeddings in Qdrant.
        
        Args:
            sentences: List of Sentence objects with embeddings
            collection_name: Target collection (uses default if None)
            
        Returns:
            Number of points stored
        """
        collection = collection_name or self.default_collection
        
        # Ensure collection exists
        if not self.collection_exists(collection):
            self.create_collection(collection)
        
        # Prepare points for storage
        points = []
        for sentence in sentences:
            if not sentence.embedding:
                raise ValueError(f"Sentence {sentence.id} has no embedding")
            
            point = PointStruct(
                id=sentence.id,
                vector=sentence.embedding,
                payload={
                    'text': sentence.text,
                    'archetype': sentence.archetype,
                    'dimensions': sentence.dimensions,
                    'contextualized_text': sentence.contextualized_text,
                    'cluster_id': sentence.cluster_id
                }
            )
            points.append(point)
        
        # Store points
        try:
            self.client.upsert(
                collection_name=collection,
                points=points
            )
            return len(points)
        except Exception as e:
            raise Exception(f"Failed to store embeddings: {str(e)}")
    
    def extract_data(self, collection_name: Optional[str] = None, 
                    limit: Optional[int] = None,
                    filter_conditions: Optional[Dict] = None) -> List[Sentence]:
        """
        Extract data from Qdrant collection.
        
        Args:
            collection_name: Source collection (uses default if None)
            limit: Maximum number of points to retrieve
            filter_conditions: Filter conditions for retrieval
            
        Returns:
            List of Sentence objects
        """
        collection = collection_name or self.default_collection
        
        try:
            # Build filter if provided
            filter_obj = None
            if filter_conditions:
                # This is a simplified filter implementation
                # Can be extended based on specific needs
                pass
            
            # Retrieve points
            points, _ = self.client.scroll(
                collection_name=collection,
                limit=limit,
                with_payload=True,
                with_vectors=True,
                scroll_filter=filter_obj
            )
            
            # Convert to Sentence objects
            sentences = []
            for point in points:
                sentence = Sentence(
                    id=str(point.id),
                    text=point.payload.get('text', ''),
                    archetype=point.payload.get('archetype', ''),
                    dimensions=point.payload.get('dimensions', []),
                    contextualized_text=point.payload.get('contextualized_text'),
                    embedding=point.vector,
                    cluster_id=point.payload.get('cluster_id')
                )
                sentences.append(sentence)
            
            return sentences
            
        except Exception as e:
            raise Exception(f"Failed to extract data from collection '{collection}': {str(e)}")
    
    def update_cluster_labels(self, sentences: List[Sentence], 
                            collection_name: Optional[str] = None) -> int:
        """
        Update cluster labels for stored sentences.
        
        Args:
            sentences: List of sentences with cluster_id populated
            collection_name: Target collection (uses default if None)
            
        Returns:
            Number of points updated
        """
        collection = collection_name or self.default_collection
        
        updated_count = 0
        for sentence in sentences:
            if sentence.cluster_id is not None:
                try:
                    self.client.set_payload(
                        collection_name=collection,
                        payload={'cluster_id': sentence.cluster_id},
                        points=[sentence.id]
                    )
                    updated_count += 1
                except Exception as e:
                    print(f"Failed to update cluster label for point {sentence.id}: {str(e)}")
        
        return updated_count
    
    def search_similar(self, query_vector: List[float], 
                      collection_name: Optional[str] = None,
                      limit: int = 10) -> List[Sentence]:
        """
        Search for similar vectors in the collection.
        
        Args:
            query_vector: Query embedding vector
            collection_name: Target collection (uses default if None)
            limit: Maximum number of results
            
        Returns:
            List of similar Sentence objects
        """
        collection = collection_name or self.default_collection
        
        try:
            results = self.client.search(
                collection_name=collection,
                query_vector=query_vector,
                limit=limit,
                with_payload=True
            )
            
            sentences = []
            for result in results:
                sentence = Sentence(
                    id=str(result.id),
                    text=result.payload.get('text', ''),
                    archetype=result.payload.get('archetype', ''),
                    dimensions=result.payload.get('dimensions', []),
                    contextualized_text=result.payload.get('contextualized_text'),
                    embedding=None,  # Not returned in search results
                    cluster_id=result.payload.get('cluster_id')
                )
                sentences.append(sentence)
            
            return sentences
            
        except Exception as e:
            raise Exception(f"Failed to search collection '{collection}': {str(e)}")
