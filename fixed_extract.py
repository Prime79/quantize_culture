#!/usr/bin/env python3
"""
Fixed version of data extraction to avoid hanging queries.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from qdrant_client import QdrantClient

def extract_data_fixed(collection_name="company_culture_embeddings", limit=None):
    """
    Fixed version of extract_data that won't hang.
    """
    print(f"🔄 Extracting data from collection '{collection_name}' (FIXED VERSION)...")
    
    client = QdrantClient(host="localhost", port=6333)
    
    # Safety limits to prevent infinite loops
    max_batches = 20  # Maximum number of batches to prevent infinite loop
    batch_size = 50   # Smaller batch size for reliability
    
    points = []
    offset = None
    batch_count = 0
    
    while batch_count < max_batches:
        print(f"   Fetching batch {batch_count + 1}/{max_batches} (offset: {str(offset)[:36] if offset else 'None'}...)")
        
        try:
            result = client.scroll(
                collection_name=collection_name,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=True
            )
            
            batch_points, next_offset = result
            
            if not batch_points:  # No more points
                print(f"   ✅ No more points returned - finished")
                break
                
            points.extend(batch_points)
            print(f"   ✅ Got {len(batch_points)} points (total: {len(points)})")
            
            # Check limit
            if limit and len(points) >= limit:
                points = points[:limit]
                print(f"   ✅ Reached limit of {limit} points")
                break
            
            # Update offset
            offset = next_offset
            batch_count += 1
            
            # Check if Qdrant indicates no more data
            if next_offset is None:
                print(f"   ✅ Qdrant indicates no more data (offset=None)")
                break
                
        except Exception as e:
            print(f"   ❌ Error in batch {batch_count + 1}: {e}")
            break
    
    if batch_count >= max_batches:
        print(f"   ⚠️  Reached maximum batch limit ({max_batches}) - possible infinite loop prevented")
    
    print(f"📊 Extracted {len(points)} points total")
    
    if not points:
        print("❌ No points extracted")
        return None, None
    
    # Convert to DataFrame
    print("🔄 Converting to DataFrame...")
    data_rows = []
    embeddings = []
    
    for i, point in enumerate(points):
        if i < 5:  # Show progress for first few
            print(f"   Processing point {i+1}: {point.id}")
        row = {
            'point_id': point.id,
            'sentence': point.payload.get('sentence', ''),
            'category': point.payload.get('category', ''),
            'subcategory': point.payload.get('subcategory', ''),
            'company': point.payload.get('company', ''),
            'rating': point.payload.get('rating', None),
            'source': point.payload.get('source', ''),
            'timestamp': point.payload.get('timestamp', '')
        }
        data_rows.append(row)
        
        if point.vector:
            embeddings.append(point.vector)
        else:
            print(f"   ⚠️  Point {point.id} has no vector")
    
    df = pd.DataFrame(data_rows)
    embeddings_array = np.array(embeddings) if embeddings else None
    
    print(f"✅ Data shape: {df.shape}")
    if embeddings_array is not None:
        print(f"✅ Embeddings shape: {embeddings_array.shape}")
    else:
        print("❌ No embeddings extracted")
    
    return df, embeddings_array

if __name__ == "__main__":
    # Test the fixed extraction
    print("🧪 Testing Fixed Data Extraction")
    print("=" * 40)
    
    # Test with all data
    df, embeddings = extract_data_fixed(limit=None)
    
    if df is not None:
        print(f"\\n📊 SUCCESS: Extracted {len(df)} records")
        print(f"Columns: {list(df.columns)}")
        print("\\nSample data:")
        print(df[['sentence', 'category']].head(3))
    else:
        print("\\n❌ FAILED: Could not extract data")
