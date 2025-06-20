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
                # Handle different payload structures
                # New structure: 'text', 'archetype', 'cluster_id'
                # Legacy structure: 'sentence', 'cluster_name', 'cluster'
                
                text = point.payload.get('text') or point.payload.get('sentence', '')
                archetype = point.payload.get('archetype') or point.payload.get('cluster_name', '')
                cluster_id = point.payload.get('cluster_id') or point.payload.get('cluster')
                
                sentence = Sentence(
                    id=str(point.id),
                    text=text,
                    archetype=archetype,
                    dimensions=point.payload.get('dimensions', []),
                    contextualized_text=point.payload.get('contextualized_text'),
                    embedding=point.vector,
                    cluster_id=cluster_id
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
    
    def store_sentences_with_dl_metadata(self, sentences: List[Sentence], 
                                       collection_name: Optional[str] = None) -> int:
        """
        Store sentences with complete DL metadata in Qdrant.
        
        Args:
            sentences: List of sentences with DL metadata
            collection_name: Target collection name
            
        Returns:
            Number of sentences stored
        """
        collection = collection_name or self.default_collection
        
        try:
            # Prepare points with DL metadata
            points = []
            for sentence in sentences:
                if not sentence.embedding:
                    raise ValueError(f"Sentence {sentence.id} missing embedding vector")
                
                payload = {
                    'text': sentence.text,
                    'archetype': sentence.archetype,
                    'dimensions': sentence.dimensions,
                    'contextualized_text': sentence.contextualized_text,
                    'cluster_id': sentence.cluster_id,
                    'dl_category': sentence.dl_category,
                    'dl_subcategory': sentence.dl_subcategory,
                    'dl_archetype': sentence.dl_archetype,
                    'actual_phrase': sentence.actual_phrase
                }
                
                # Remove None values
                payload = {k: v for k, v in payload.items() if v is not None}
                
                point = {
                    'id': sentence.id,
                    'vector': sentence.embedding,
                    'payload': payload
                }
                points.append(point)
            
            # Store in batches
            batch_size = 100
            total_stored = 0
            
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.client.upsert(
                    collection_name=collection,
                    points=batch
                )
                total_stored += len(batch)
            
            return total_stored
            
        except Exception as e:
            raise Exception(f"Failed to store sentences with DL metadata: {str(e)}")
    
    def query_by_dl_metadata(self, collection_name: Optional[str] = None,
                           dl_category: Optional[str] = None,
                           dl_subcategory: Optional[str] = None,
                           dl_archetype: Optional[str] = None,
                           limit: int = 100) -> List[Sentence]:
        """
        Query sentences by DL metadata.
        
        Args:
            collection_name: Source collection
            dl_category: Filter by DL category
            dl_subcategory: Filter by DL subcategory
            dl_archetype: Filter by DL archetype
            limit: Maximum number of results
            
        Returns:
            List of matching sentences
        """
        collection = collection_name or self.default_collection
        
        try:
            # Build filter conditions
            filter_conditions = {}
            if dl_category:
                filter_conditions['dl_category'] = dl_category
            if dl_subcategory:
                filter_conditions['dl_subcategory'] = dl_subcategory
            if dl_archetype:
                filter_conditions['dl_archetype'] = dl_archetype
            
            # Build Qdrant filter
            filter_obj = None
            if filter_conditions:
                must_conditions = []
                for field, value in filter_conditions.items():
                    must_conditions.append({
                        "key": field,
                        "match": {"value": value}
                    })
                
                filter_obj = {"must": must_conditions}
            
            # Execute search
            points, _ = self.client.scroll(
                collection_name=collection,
                scroll_filter=filter_obj,
                limit=limit,
                with_payload=True,
                with_vectors=True
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
                    cluster_id=point.payload.get('cluster_id'),
                    dl_category=point.payload.get('dl_category'),
                    dl_subcategory=point.payload.get('dl_subcategory'),
                    dl_archetype=point.payload.get('dl_archetype'),
                    actual_phrase=point.payload.get('actual_phrase')
                )
                sentences.append(sentence)
            
            return sentences
            
        except Exception as e:
            raise Exception(f"Failed to query by DL metadata: {str(e)}")
    
    def validate_dl_metadata_completeness(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate that all points in collection have complete DL metadata.
        
        Args:
            collection_name: Collection to validate
            
        Returns:
            Validation report
        """
        collection = collection_name or self.default_collection
        
        try:
            # Get all points
            points, _ = self.client.scroll(
                collection_name=collection,
                limit=10000,  # Adjust based on collection size
                with_payload=True,
                with_vectors=False
            )
            
            total_points = len(points)
            complete_metadata = 0
            missing_fields = {'dl_category': 0, 'dl_subcategory': 0, 'dl_archetype': 0}
            
            for point in points:
                payload = point.payload
                
                has_category = payload.get('dl_category') is not None and payload.get('dl_category') != ""
                has_subcategory = payload.get('dl_subcategory') is not None and payload.get('dl_subcategory') != ""
                has_archetype = payload.get('dl_archetype') is not None and payload.get('dl_archetype') != ""
                
                if not has_category:
                    missing_fields['dl_category'] += 1
                if not has_subcategory:
                    missing_fields['dl_subcategory'] += 1
                if not has_archetype:
                    missing_fields['dl_archetype'] += 1
                
                if has_category and has_subcategory and has_archetype:
                    complete_metadata += 1
            
            completion_rate = complete_metadata / total_points if total_points > 0 else 0
            
            return {
                'total_points': total_points,
                'complete_metadata': complete_metadata,
                'completion_rate': completion_rate,
                'missing_fields': missing_fields,
                'is_complete': completion_rate == 1.0
            }
            
        except Exception as e:
            raise Exception(f"Failed to validate DL metadata completeness: {str(e)}")
