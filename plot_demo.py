#!/usr/bin/env python3
"""
Demo script for plotting UMAP clustering results.
Creates beautiful visualizations of the company culture data clusters.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.extract import DataExtractorAnalyzer
import matplotlib.pyplot as plt

def create_2d_plot():
    """Create a 2D UMAP cluster plot."""
    print("🎨 Creating 2D UMAP cluster visualization...")
    
    analyzer = DataExtractorAnalyzer()
    
    # Extract and process data
    analyzer.extract_data(limit=100)  # Use 100 points for demo
    analyzer.reduce_dimensions(n_components=2, n_neighbors=15, min_dist=0.1)
    analyzer.cluster_data(min_cluster_size=5)
    
    # Create the plot
    analyzer.plot_clusters(
        figsize=(12, 8),
        save_path="umap_2d_clusters.png",
        show_labels=True,
        alpha=0.7
    )
    
    # Print cluster summary
    summary = analyzer.get_cluster_summary()
    print("\n📊 2D Clustering Summary:")
    for _, row in summary.head().iterrows():
        print(f"   {row['cluster_name']}: {row['size']} points")
        if row['sample_sentences']:
            print(f"      Sample: \"{row['sample_sentences'][0][:80]}...\"")
    
    return analyzer

def create_3d_plot():
    """Create a 3D UMAP cluster plot."""
    print("\n🎨 Creating 3D UMAP cluster visualization...")
    
    analyzer = DataExtractorAnalyzer()
    
    # Extract and process data
    analyzer.extract_data(limit=100)  # Use 100 points for demo
    analyzer.reduce_dimensions(n_components=3, n_neighbors=15, min_dist=0.1)
    analyzer.cluster_data(min_cluster_size=5)
    
    # Create the plot
    analyzer.plot_clusters(
        figsize=(12, 8),
        save_path="umap_3d_clusters.png",
        show_labels=True,
        alpha=0.7
    )
    
    # Print cluster summary
    summary = analyzer.get_cluster_summary()
    print("\n📊 3D Clustering Summary:")
    for _, row in summary.head().iterrows():
        print(f"   {row['cluster_name']}: {row['size']} points")
        if row['sample_sentences']:
            print(f"      Sample: \"{row['sample_sentences'][0][:80]}...\"")
    
    return analyzer

def create_comparison_plot():
    """Create a side-by-side comparison of 2D and 3D plots."""
    print("\n🎨 Creating 2D vs 3D comparison plot...")
    
    analyzer = DataExtractorAnalyzer()
    analyzer.extract_data(limit=100)
    
    # Create comparison plot
    analyzer.plot_cluster_comparison(
        figsize=(16, 6),
        save_path="umap_comparison.png"
    )
    
    print("   ✅ Comparison plot created!")

def analyze_all_data():
    """Analyze and plot all available data."""
    print("\n🎨 Creating plot with ALL data...")
    
    analyzer = DataExtractorAnalyzer()
    
    # Extract all data
    data = analyzer.extract_data()  # No limit = all data
    print(f"   Processing {len(data)} total data points...")
    
    # Reduce dimensions and cluster
    analyzer.reduce_dimensions(n_components=2, n_neighbors=20, min_dist=0.05)
    analyzer.cluster_data(min_cluster_size=8)
    
    # Create detailed plot
    analyzer.plot_clusters(
        figsize=(14, 10),
        save_path="umap_all_data.png",
        show_labels=True,
        alpha=0.6
    )
    
    # Detailed summary
    summary = analyzer.get_cluster_summary()
    print(f"\n📊 Complete Dataset Analysis:")
    print(f"   Total sentences: {len(data)}")
    print(f"   Number of clusters: {len(summary[summary['cluster_id'] >= 0])}")
    print(f"   Noise points: {len(summary[summary['cluster_id'] == -1])}")
    
    print("\n📝 Cluster Details:")
    for _, row in summary.iterrows():
        if row['cluster_id'] >= 0:  # Skip noise for detailed view
            print(f"\n   🔍 {row['cluster_name'].upper()}: {row['size']} sentences")
            # Show top 3 sample sentences
            for i, sentence in enumerate(row['sample_sentences'][:3], 1):
                print(f"      {i}. \"{sentence[:100]}...\"")
    
    return analyzer

def main():
    """Run all plotting demonstrations."""
    print("🚀 Starting UMAP Clustering Visualization Demo\n")
    
    try:
        # Create individual plots
        analyzer_2d = create_2d_plot()
        analyzer_3d = create_3d_plot()
        
        # Create comparison plot
        create_comparison_plot()
        
        # Analyze all data
        analyzer_all = analyze_all_data()
        
        print(f"\n🎉 All visualizations completed!")
        print(f"📁 Generated files:")
        print(f"   • umap_2d_clusters.png")
        print(f"   • umap_3d_clusters.png") 
        print(f"   • umap_comparison.png")
        print(f"   • umap_all_data.png")
        
        print(f"\n💡 Next steps:")
        print(f"   • Examine the cluster plots to understand data groupings")
        print(f"   • Analyze cluster themes from the sample sentences")
        print(f"   • Consider adjusting UMAP/HDBSCAN parameters for different views")
        
    except Exception as e:
        print(f"❌ Error during visualization: {str(e)}")
        raise

if __name__ == "__main__":
    main()
