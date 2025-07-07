#!/usr/bin/env python3
"""
UMAP dimensionality reduction for embeddings in Qdrant.
Reduces 1536-dimensional embeddings to 10 dimensions and updates the records.
"""

import numpy as np
import umap
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from typing import List, Dict, Any
import time
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

def extract_embeddings_from_qdrant(collection_name: str = "target_test") -> tuple:
    """
    Extract all embeddings and metadata from Qdrant collection.
    
    Args:
        collection_name: Name of the Qdrant collection
        
    Returns:
        tuple: (embeddings_array, point_ids, payloads)
    """
    print(f"📊 Extracting embeddings from collection '{collection_name}'...")
    
    # Connect to Qdrant
    client = QdrantClient(host="localhost", port=6333)
    
    # Get all points with vectors
    points, _ = client.scroll(
        collection_name=collection_name,
        limit=10000,  # Get all points
        with_payload=True,
        with_vectors=True
    )
    
    print(f"✅ Retrieved {len(points)} points from Qdrant")
    
    # Extract embeddings, IDs, and payloads
    embeddings = []
    point_ids = []
    payloads = []
    
    for point in points:
        if hasattr(point, 'vector') and point.vector:
            embeddings.append(point.vector)
            point_ids.append(point.id)
            payloads.append(point.payload)
    
    embeddings_array = np.array(embeddings)
    print(f"📈 Embeddings shape: {embeddings_array.shape}")
    
    return embeddings_array, point_ids, payloads

def reduce_dimensions_with_umap(embeddings: np.ndarray, 
                               n_components: int = 10,
                               n_neighbors: int = 15,
                               min_dist: float = 0.1,
                               random_state: int = 42) -> np.ndarray:
    """
    Reduce embedding dimensions using UMAP.
    
    Args:
        embeddings: Array of high-dimensional embeddings
        n_components: Target number of dimensions (default: 10)
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        random_state: Random state for reproducibility
        
    Returns:
        np.ndarray: Reduced embeddings
    """
    print(f"🎯 Reducing {embeddings.shape[1]}D embeddings to {n_components}D using UMAP...")
    print(f"   Parameters: n_neighbors={n_neighbors}, min_dist={min_dist}")
    
    # Initialize UMAP
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        metric='cosine'  # Good for text embeddings
    )
    
    # Fit and transform
    start_time = time.time()
    reduced_embeddings = reducer.fit_transform(embeddings)
    end_time = time.time()
    
    print(f"✅ UMAP reduction complete in {end_time - start_time:.2f} seconds")
    print(f"📊 Reduced embeddings shape: {reduced_embeddings.shape}")
    
    return reduced_embeddings

def update_qdrant_with_umap(point_ids: List[str], 
                           payloads: List[Dict[str, Any]], 
                           reduced_embeddings: np.ndarray,
                           collection_name: str = "target_test") -> None:
    """
    Update Qdrant records with UMAP-reduced embeddings.
    
    Args:
        point_ids: List of point IDs
        payloads: List of original payloads
        reduced_embeddings: UMAP-reduced embeddings
        collection_name: Name of the Qdrant collection
    """
    print(f"💾 Updating Qdrant collection '{collection_name}' with UMAP embeddings...")
    
    # Connect to Qdrant
    client = QdrantClient(host="localhost", port=6333)
    
    # Prepare updated points
    updated_points = []
    for i, (point_id, payload, umap_embedding) in enumerate(zip(point_ids, payloads, reduced_embeddings)):
        # Create updated payload with UMAP embedding
        updated_payload = payload.copy()
        updated_payload['umap_embedding'] = umap_embedding.tolist()
        updated_payload['umap_dimensions'] = len(umap_embedding)
        updated_payload['umap_created_at'] = time.time()
        
        # Note: We keep the original 1536D vector as the main vector
        # and add UMAP as additional metadata
        updated_points.append(PointStruct(
            id=point_id,
            vector=payload.get('original_vector', [0] * 1536),  # Keep original if available
            payload=updated_payload
        ))
        
        if (i + 1) % 100 == 0:
            print(f"   📤 Prepared {i + 1}/{len(point_ids)} points...")
    
    print(f"✅ Prepared {len(updated_points)} points for update")
    
    # Batch update points
    print(f"📤 Updating points in Qdrant...")
    client.upsert(collection_name=collection_name, points=updated_points)
    
    print(f"✅ Successfully updated {len(updated_points)} points with UMAP embeddings")

def create_umap_only_collection(point_ids: List[str], 
                               payloads: List[Dict[str, Any]], 
                               reduced_embeddings: np.ndarray,
                               collection_name: str = "target_test_umap10d") -> None:
    """
    Create a new collection with UMAP embeddings as the main vectors.
    
    Args:
        point_ids: List of point IDs
        payloads: List of original payloads
        reduced_embeddings: UMAP-reduced embeddings
        collection_name: Name for the new Qdrant collection
    """
    print(f"🆕 Creating new collection '{collection_name}' with 10D UMAP vectors...")
    
    from qdrant_client.models import Distance, VectorParams
    
    # Connect to Qdrant
    client = QdrantClient(host="localhost", port=6333)
    
    # Create new collection with 10D vectors
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=10, distance=Distance.COSINE)
        )
        print(f"✅ Created collection '{collection_name}' with 10D vectors")
    except Exception as e:
        print(f"⚠️  Collection might already exist: {e}")
    
    # Prepare points with UMAP vectors
    umap_points = []
    for i, (point_id, payload, umap_embedding) in enumerate(zip(point_ids, payloads, reduced_embeddings)):
        # Create payload with both original and UMAP info
        umap_payload = payload.copy()
        umap_payload['original_dimensions'] = 1536
        umap_payload['umap_dimensions'] = 10
        umap_payload['umap_created_at'] = time.time()
        umap_payload['reduction_method'] = 'UMAP'
        
        umap_points.append(PointStruct(
            id=point_id,
            vector=umap_embedding.tolist(),  # Use UMAP as main vector
            payload=umap_payload
        ))
        
        if (i + 1) % 100 == 0:
            print(f"   📤 Prepared {i + 1}/{len(point_ids)} UMAP points...")
    
    print(f"✅ Prepared {len(umap_points)} points for UMAP collection")
    
    # Insert points into new collection
    print(f"📤 Inserting points into UMAP collection...")
    client.upsert(collection_name=collection_name, points=umap_points)
    
    print(f"✅ Successfully created collection '{collection_name}' with {len(umap_points)} 10D vectors")
    
    # Verify collection
    collection_info = client.get_collection(collection_name)
    print(f"📊 UMAP collection contains {collection_info.points_count} total points")

def main():
    """Main function to reduce embeddings and update Qdrant."""
    print("🎯 UMAP Dimensionality Reduction for Qdrant Embeddings")
    print("=" * 60)
    
    try:
        # Step 1: Extract embeddings from Qdrant
        embeddings, point_ids, payloads = extract_embeddings_from_qdrant("target_test")
        
        if len(embeddings) == 0:
            print("❌ No embeddings found in collection")
            return
        
        # Step 2: Reduce dimensions with UMAP
        reduced_embeddings = reduce_dimensions_with_umap(
            embeddings, 
            n_components=10,
            n_neighbors=15,
            min_dist=0.1
        )
        
        # Step 3: Create new collection with UMAP vectors
        create_umap_only_collection(point_ids, payloads, reduced_embeddings)
        
        # Step 4: Also update original collection with UMAP metadata
        print(f"\n💾 Adding UMAP embeddings as metadata to original collection...")
        
        # We need to extract original vectors first for proper update
        client = QdrantClient(host="localhost", port=6333)
        original_points, _ = client.scroll(
            collection_name="target_test",
            limit=10000,
            with_payload=True,
            with_vectors=True
        )
        
        # Update with UMAP metadata while preserving original vectors
        updated_points = []
        for i, (point, umap_embedding) in enumerate(zip(original_points, reduced_embeddings)):
            updated_payload = point.payload.copy()
            updated_payload['umap_embedding'] = umap_embedding.tolist()
            updated_payload['umap_dimensions'] = 10
            updated_payload['umap_created_at'] = time.time()
            
            updated_points.append(PointStruct(
                id=point.id,
                vector=point.vector,  # Keep original 1536D vector
                payload=updated_payload
            ))
        
        client.upsert(collection_name="target_test", points=updated_points)
        print(f"✅ Updated original collection with UMAP metadata")
        
        print(f"\n🎉 UMAP dimensionality reduction completed!")
        print(f"📊 Processed {len(embeddings)} embeddings")
        print(f"🔽 Reduced from 1536D to 10D")
        print(f"💾 Collections:")
        print(f"   • target_test: Original 1536D + UMAP metadata")
        print(f"   • target_test_umap10d: New collection with 10D UMAP vectors")
        
        # Show sample of reduced embedding
        print(f"\n📈 Sample UMAP embedding (first 10 values):")
        print(f"   {reduced_embeddings[0]}")
        
    except Exception as e:
        print(f"❌ Error in UMAP processing: {e}")
        raise

if __name__ == "__main__":
    main()
