#!/usr/bin/env python3
"""
Debug what's causing the extract_data hang
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from qdrant_client import QdrantClient

def debug_extract_hang():
    """Debug what's causing the extract_data hang"""
    
    client = QdrantClient(host="localhost", port=6333)
    collection_name = "culture_original_test"
    
    print(f"🔍 DEBUGGING EXTRACT HANG for collection: {collection_name}")
    
    # Check if collection exists
    try:
        collections = client.get_collections()
        print(f"Available collections: {[c.name for c in collections.collections]}")
        
        if collection_name not in [c.name for c in collections.collections]:
            print(f"❌ Collection {collection_name} doesn't exist!")
            return
            
        # Check collection info
        info = client.get_collection(collection_name)
        print(f"Collection info: points_count={info.points_count}")
        
        if info.points_count == 0:
            print("❌ Collection is empty!")
            return
            
        # Test basic scroll - this is where it probably hangs
        print("Testing basic scroll...")
        
        # Start with small limit to see if it works
        scroll_result = client.scroll(
            collection_name=collection_name,
            limit=10,
            with_payload=True,
            with_vectors=False  # Start without vectors
        )
        
        print(f"Basic scroll worked: {len(scroll_result[0])} points returned")
        print(f"Next page offset: {scroll_result[1]}")
        
        # Now test with vectors - this might be the issue
        print("Testing scroll with vectors...")
        
        scroll_result_vectors = client.scroll(
            collection_name=collection_name,
            limit=10,
            with_payload=True,
            with_vectors=True  # This might hang
        )
        
        print(f"Vector scroll worked: {len(scroll_result_vectors[0])} points returned")
        
        # Test the scroll continuation - this is likely where the infinite loop is
        print("Testing scroll continuation...")
        
        all_points = []
        next_offset = None
        iterations = 0
        max_iterations = 10  # Safety limit
        
        while iterations < max_iterations:
            print(f"Iteration {iterations + 1}")
            
            scroll_result = client.scroll(
                collection_name=collection_name,
                offset=next_offset,
                limit=50,
                with_payload=True,
                with_vectors=True
            )
            
            points, next_offset = scroll_result
            all_points.extend(points)
            
            print(f"  Got {len(points)} points, next_offset: {next_offset}")
            
            # THIS IS THE BUG - check the exit condition
            if next_offset is None:
                print("  ✅ next_offset is None - should exit")
                break
            elif len(points) == 0:
                print("  ✅ No more points - should exit")
                break
            else:
                print(f"  ➡️  Continue scrolling...")
            
            iterations += 1
        
        if iterations >= max_iterations:
            print("❌ Hit max iterations - would be infinite loop!")
        else:
            print(f"✅ Scroll completed normally in {iterations + 1} iterations")
            
        print(f"Total points collected: {len(all_points)}")
        
    except Exception as e:
        print(f"❌ Error during debug: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_extract_hang()
