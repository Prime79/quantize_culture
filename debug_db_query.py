#!/usr/bin/env python3
"""
Debug script to test the database query issue.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from qdrant_client import QdrantClient

def debug_database_query():
    """Debug the database query to find the hanging issue."""
    print("🔍 Debugging Database Query Issue")
    print("=" * 40)
    
    # Connect to Qdrant
    client = QdrantClient(host="localhost", port=6333)
    collection_name = "company_culture_embeddings"
    
    try:
        # First check if collection exists
        print("📊 Checking collection...")
        collections = client.get_collections()
        collection_names = [c.name for c in collections.collections]
        print(f"Available collections: {collection_names}")
        
        if collection_name not in collection_names:
            print(f"❌ Collection '{collection_name}' not found!")
            return
        
        # Get collection info
        collection_info = client.get_collection(collection_name)
        print(f"✅ Collection found: {collection_info.points_count} points")
        
        # Test small scroll
        print("\n🔄 Testing small scroll (limit=10)...")
        result = client.scroll(
            collection_name=collection_name,
            limit=10,
            with_payload=True,
            with_vectors=False  # Don't get vectors for this test
        )
        points, next_offset = result
        print(f"✅ Got {len(points)} points, next_offset: {next_offset}")
        
        # Test larger scroll
        print("\n🔄 Testing larger scroll (limit=100)...")
        result = client.scroll(
            collection_name=collection_name,
            limit=100,
            with_payload=True,
            with_vectors=False
        )
        points, next_offset = result
        print(f"✅ Got {len(points)} points, next_offset: {next_offset}")
        
        # Test problematic scroll - get ALL points step by step
        print("\n🔄 Testing full scroll with timeout...")
        all_points = []
        offset = None
        batch_count = 0
        max_batches = 10  # Safety limit
        
        while batch_count < max_batches:
            print(f"   Batch {batch_count + 1}: offset={offset}")
            
            result = client.scroll(
                collection_name=collection_name,
                limit=50,  # Smaller batch size
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            
            points, next_offset = result
            print(f"   Got {len(points)} points, next_offset: {next_offset}")
            
            if not points:  # No more points
                print("   ✅ No more points - done")
                break
                
            all_points.extend(points)
            offset = next_offset
            batch_count += 1
            
            if next_offset is None:  # Qdrant indicates no more data
                print("   ✅ Next offset is None - done")
                break
        
        print(f"\n📊 Total points collected: {len(all_points)}")
        
        if batch_count >= max_batches:
            print("⚠️  Stopped at safety limit - possible infinite loop!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_database_query()
