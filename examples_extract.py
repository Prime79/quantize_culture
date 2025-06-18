#!/usr/bin/env python3
"""
Individual function examples for extract.py
Shows how to use each of the four main functionalities separately.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.extract import DataExtractorAnalyzer

def example_1_extract_all():
    """Example 1: Extract all data from database"""
    print("Example 1: Extract ALL data from database")
    analyzer = DataExtractorAnalyzer()
    data = analyzer.extract_data()  # No limit = all data
    print(f"Total points extracted: {len(data)}")
    print(f"Columns: {list(data.columns)}")
    print()

def example_2_extract_filtered():
    """Example 2: Extract specific data with filters"""
    print("Example 2: Extract data with filters")
    analyzer = DataExtractorAnalyzer()
    
    # Example filter: only sentences from a specific category
    # Note: This will only work if your data has these fields
    filters = {
        # "category": "Leadership",  # Uncomment if you have categories
        # "rating_min": 3           # Uncomment if you have ratings
    }
    
    data = analyzer.extract_data(limit=20, filter_conditions=filters if filters else None)
    print(f"Filtered points extracted: {len(data)}")
    print(f"Sample sentences:")
    for i, sentence in enumerate(data['sentence'].head(3)):
        print(f"  {i+1}. {sentence[:100]}...")
    print()

def example_3_umap_3d():
    """Example 3: 3D UMAP reduction"""
    print("Example 3: 3D UMAP dimensionality reduction")
    analyzer = DataExtractorAnalyzer()
    analyzer.extract_data(limit=30)  # Small sample for demo
    
    # 3D reduction
    reduced = analyzer.reduce_dimensions(
        n_components=3,
        n_neighbors=10,
        min_dist=0.05,
        metric='cosine'
    )
    
    print(f"Original dimensions: {analyzer.embeddings.shape[1]}")
    print(f"Reduced to: {reduced.shape[1]} dimensions")
    print(f"3D coordinates sample:")
    for i in range(min(3, len(reduced))):
        x, y, z = reduced[i]
        print(f"  Point {i+1}: ({x:.3f}, {y:.3f}, {z:.3f})")
    print()

def example_4_custom_clustering():
    """Example 4: Custom HDBSCAN clustering parameters"""
    print("Example 4: Custom HDBSCAN clustering")
    analyzer = DataExtractorAnalyzer()
    analyzer.extract_data(limit=50)
    analyzer.reduce_dimensions(n_components=2)
    
    # Custom clustering parameters
    clusters = analyzer.cluster_data(
        min_cluster_size=8,      # Larger minimum cluster size
        min_samples=4,           # More samples required
        cluster_selection_epsilon=0.1,  # Some epsilon for cluster selection
        metric='euclidean'
    )
    
    summary = analyzer.get_cluster_summary()
    print("Custom clustering results:")
    for _, row in summary.iterrows():
        print(f"  {row['cluster_name']}: {row['size']} points")
    print()

def example_5_store_without_running_analysis():
    """Example 5: Store pre-computed cluster labels"""
    print("Example 5: Store custom cluster labels")
    analyzer = DataExtractorAnalyzer()
    data = analyzer.extract_data(limit=10)
    
    # Simulate some custom cluster labels (e.g., from external analysis)
    import numpy as np
    analyzer.cluster_labels = np.random.randint(-1, 3, size=len(data))  # Random clusters
    analyzer.data['cluster'] = analyzer.cluster_labels
    
    # Store to database
    updated = analyzer.store_clusters_to_database()
    print(f"Stored {updated} custom cluster labels to database")
    print()

if __name__ == "__main__":
    print("=== Extract.py Individual Function Examples ===\n")
    
    # Run examples (comment out any you don't want to run)
    example_1_extract_all()
    example_2_extract_filtered()
    example_3_umap_3d()
    example_4_custom_clustering()
    # example_5_store_without_running_analysis()  # Commented out to avoid overwriting
    
    print("✅ All examples completed!")
