#!/usr/bin/env python3
"""
Simple script to create a 2D UMAP plot of the current data.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.extract import DataExtractorAnalyzer

def create_simple_plot():
    """Create a simple 2D UMAP plot."""
    print("🎨 Creating simple 2D UMAP cluster plot...")
    
    analyzer = DataExtractorAnalyzer()
    
    # Extract and process data (all data)
    print("📊 Extracting data...")
    analyzer.extract_data(limit=200)  # Reasonable limit for visualization
    
    print("🎯 Reducing dimensions to 2D...")
    analyzer.reduce_dimensions(n_components=2, n_neighbors=15, min_dist=0.1)
    
    print("🔍 Clustering with HDBSCAN...")
    analyzer.cluster_data(min_cluster_size=8)
    
    # Create just the 2D plot
    print("📈 Creating 2D plot...")
    analyzer.plot_clusters(
        figsize=(14, 10),
        save_path="company_culture_clusters.png",
        title="Company Culture Sentence Clusters (UMAP + HDBSCAN)",
        alpha=0.8
    )
    
    # Show summary
    summary = analyzer.get_cluster_summary()
    print(f"\n📊 Analysis Results:")
    print(f"   Total sentences analyzed: {len(analyzer.data)}")
    print(f"   Clusters identified: {len(summary[summary['cluster_id'] >= 0])}")
    print(f"   Noise points: {sum(summary['cluster_id'] == -1)}")
    
    print(f"\n📝 Cluster Themes:")
    for _, row in summary.iterrows():
        if row['cluster_id'] >= 0:
            print(f"\n   🎯 {row['cluster_name'].upper()} ({row['size']} sentences):")
            # Show representative sentences
            for i, sentence in enumerate(row['sample_sentences'][:2], 1):
                print(f"      • \"{sentence[:90]}...\"")
    
    print(f"\n✅ Plot saved as 'company_culture_clusters.png'")
    return analyzer

if __name__ == "__main__":
    create_simple_plot()
