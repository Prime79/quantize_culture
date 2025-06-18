#!/usr/bin/env python3
"""
Demo script for plotting UMAP clustering results using utils.py functions.
Creates beautiful visualizations of the company culture data clusters.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.extract import DataExtractorAnalyzer

def create_sample_plots():
    """Create sample plots with a subset of data."""
    print("🎨 Creating sample UMAP cluster visualizations...")
    
    analyzer = DataExtractorAnalyzer()
    
    # Extract and process data
    print("📊 Extracting data...")
    analyzer.extract_data(limit=100)  # Use 100 points for demo
    
    print("🎯 Reducing dimensions...")
    analyzer.reduce_dimensions(n_components=2, n_neighbors=15, min_dist=0.1)
    
    print("🔍 Clustering...")
    analyzer.cluster_data(min_cluster_size=5)
    
    # Create all plots
    print("📈 Creating plots...")
    analyzer.create_all_plots(
        save_2d="sample_umap_2d.png",
        save_3d="sample_umap_3d.png",
        save_comparison="sample_comparison.png",
        save_summary="sample_summary.png"
    )
    
    # Print summary
    summary = analyzer.get_cluster_summary()
    print(f"\n📊 Sample Analysis Summary:")
    print(f"   Total points: {len(analyzer.data)}")
    print(f"   Clusters found: {len(summary[summary['cluster_id'] >= 0])}")
    print(f"   Noise points: {len(summary[summary['cluster_id'] == -1])}")
    
    print(f"\n📝 Cluster Details:")
    for _, row in summary.iterrows():
        if row['cluster_id'] >= 0:
            print(f"   • {row['cluster_name']}: {row['size']} points")
            if row['sample_sentences']:
                print(f"     Sample: \"{row['sample_sentences'][0][:80]}...\"")
    
    return analyzer

def create_full_analysis():
    """Create analysis with all available data."""
    print(f"\n🎨 Creating full dataset analysis...")
    
    analyzer = DataExtractorAnalyzer()
    
    # Extract all data
    print("📊 Extracting all data...")
    data = analyzer.extract_data()  # No limit = all data
    print(f"   Processing {len(data)} total data points...")
    
    # Process with parameters suitable for larger dataset
    print("🎯 Reducing dimensions...")
    analyzer.reduce_dimensions(n_components=2, n_neighbors=20, min_dist=0.05)
    
    print("🔍 Clustering...")
    analyzer.cluster_data(min_cluster_size=8, min_samples=5)
    
    # Create plots
    print("📈 Creating full analysis plots...")
    analyzer.create_all_plots(
        save_2d="full_umap_2d.png",
        save_3d="full_umap_3d.png", 
        save_comparison="full_comparison.png",
        save_summary="full_summary.png"
    )
    
    # Store results back to database
    print("💾 Storing cluster labels to database...")
    analyzer.store_clusters_to_database()
    
    # Detailed summary
    summary = analyzer.get_cluster_summary()
    print(f"\n📊 Complete Dataset Analysis:")
    print(f"   Total sentences: {len(data)}")
    print(f"   Number of clusters: {len(summary[summary['cluster_id'] >= 0])}")
    print(f"   Noise points: {sum(summary['cluster_id'] == -1)}")
    
    print(f"\n📝 Detailed Cluster Analysis:")
    for _, row in summary.iterrows():
        if row['cluster_id'] >= 0:  # Skip noise for detailed view
            print(f"\n   🔍 {row['cluster_name'].upper()}: {row['size']} sentences")
            # Show top 3 sample sentences
            for i, sentence in enumerate(row['sample_sentences'][:3], 1):
                print(f"      {i}. \"{sentence[:100]}...\"")
    
    return analyzer

def main():
    """Run the plotting demonstration."""
    print("🚀 Starting UMAP Clustering Visualization Demo\n")
    
    try:
        # Create sample plots first
        analyzer_sample = create_sample_plots()
        
        # Create full analysis
        analyzer_full = create_full_analysis()
        
        print(f"\n🎉 All visualizations completed!")
        print(f"📁 Generated files:")
        print(f"   Sample Analysis:")
        print(f"   • sample_umap_2d.png")
        print(f"   • sample_comparison.png") 
        print(f"   • sample_summary.png")
        print(f"   Full Analysis:")
        print(f"   • full_umap_2d.png")
        print(f"   • full_comparison.png")
        print(f"   • full_summary.png")
        
        print(f"\n💡 Next steps:")
        print(f"   • Examine the cluster plots to understand data groupings")
        print(f"   • Analyze cluster themes from the sample sentences")
        print(f"   • Use the stored cluster labels for further analysis")
        
    except Exception as e:
        print(f"❌ Error during visualization: {str(e)}")
        raise

if __name__ == "__main__":
    main()
