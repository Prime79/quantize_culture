#!/usr/bin/env python3
"""
Simple script to generate UMAP cluster plots without interactive display.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.extract import DataExtractorAnalyzer
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

def generate_plots():
    """Generate all cluster visualization plots."""
    print("🎨 Generating UMAP cluster visualizations...")
    
    # Create analyzer
    analyzer = DataExtractorAnalyzer()
    
    # Extract data
    print("📊 Extracting data from database...")
    analyzer.extract_data(limit=100)  # Use 100 points for demo
    print(f"   ✅ Extracted {len(analyzer.data)} data points")
    
    # 2D Analysis
    print("\n🔍 Performing 2D UMAP analysis...")
    analyzer.reduce_dimensions(n_components=2, n_neighbors=15, min_dist=0.1)
    analyzer.cluster_data(min_cluster_size=5)
    
    # Create 2D plot
    print("   📈 Creating 2D cluster plot...")
    plt.figure(figsize=(12, 8))
    analyzer.plot_clusters(save_path="umap_2d_clusters.png")
    plt.close()
    
    # Show 2D results
    summary_2d = analyzer.get_cluster_summary()
    print(f"   ✅ 2D Analysis complete:")
    print(f"      Clusters found: {len(summary_2d[summary_2d['cluster_id'] >= 0])}")
    print(f"      Noise points: {len(summary_2d[summary_2d['cluster_id'] == -1])}")
    
    # 3D Analysis
    print("\n🔍 Performing 3D UMAP analysis...")
    analyzer.reduce_dimensions(n_components=3, n_neighbors=15, min_dist=0.1)
    analyzer.cluster_data(min_cluster_size=5)
    
    # Create 3D plot
    print("   📈 Creating 3D cluster plot...")
    plt.figure(figsize=(12, 8))
    analyzer.plot_clusters(save_path="umap_3d_clusters.png")
    plt.close()
    
    # Show 3D results
    summary_3d = analyzer.get_cluster_summary()
    print(f"   ✅ 3D Analysis complete:")
    print(f"      Clusters found: {len(summary_3d[summary_3d['cluster_id'] >= 0])}")
    print(f"      Noise points: {len(summary_3d[summary_3d['cluster_id'] == -1])}")
    
    # Create comparison plot
    print("\n📊 Creating comparison plot...")
    analyzer.plot_cluster_comparison(save_path="umap_comparison.png")
    plt.close()
    
    # Print cluster details
    print(f"\n📝 Cluster Analysis Summary:")
    print(f"{'='*50}")
    
    for _, row in summary_2d.iterrows():
        if row['cluster_id'] >= 0:  # Skip noise for summary
            print(f"\n🔍 {row['cluster_name'].upper()}: {row['size']} sentences")
            # Show sample sentences
            for i, sentence in enumerate(row['sample_sentences'][:2], 1):
                print(f"   {i}. \"{sentence[:80]}...\"")
    
    return analyzer

if __name__ == "__main__":
    print("🚀 Starting UMAP Cluster Visualization Generation\n")
    
    try:
        analyzer = generate_plots()
        
        print(f"\n🎉 All visualizations generated successfully!")
        print(f"📁 Generated files:")
        print(f"   • umap_2d_clusters.png - 2D UMAP cluster visualization")
        print(f"   • umap_3d_clusters.png - 3D UMAP cluster visualization") 
        print(f"   • umap_comparison.png - Side-by-side comparison")
        
        print(f"\n💡 Next steps:")
        print(f"   • Open the PNG files to examine cluster patterns")
        print(f"   • Analyze the semantic themes within each cluster")
        print(f"   • Adjust parameters for different clustering perspectives")
        
    except Exception as e:
        print(f"❌ Error during visualization: {str(e)}")
        raise
