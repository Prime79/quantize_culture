#!/usr/bin/env python3
"""
Demo script for the extract.py functionality.
Demonstrates all four main functions: extract, reduce, cluster, and store back.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.extract import DataExtractorAnalyzer

def run_demo():
    """Run a demo of the full analysis pipeline."""
    print("🚀 Starting Company Culture Analysis Demo\n")
    
    # Initialize analyzer
    analyzer = DataExtractorAnalyzer()
    
    try:
        # 1. Extract a sample of data (limit to 50 for demo speed)
        print("📊 Step 1: Extracting sample data from database...")
        data = analyzer.extract_data(limit=50)
        print(f"   ✅ Extracted {len(data)} data points")
        print(f"   📝 Sample sentence: '{data.iloc[0]['sentence'][:100]}...'\n")
        
        # 2. Reduce dimensions with UMAP
        print("🎯 Step 2: Reducing dimensions with UMAP...")
        reduced_embeddings = analyzer.reduce_dimensions(
            n_components=2,  # 2D for demo
            n_neighbors=10,  # Smaller for smaller dataset
            min_dist=0.1
        )
        print(f"   ✅ Reduced from {analyzer.embeddings.shape[1]} to {reduced_embeddings.shape[1]} dimensions")
        print(f"   📏 Reduced shape: {reduced_embeddings.shape}\n")
        
        # 3. Cluster with HDBSCAN
        print("🔍 Step 3: Clustering with HDBSCAN...")
        cluster_labels = analyzer.cluster_data(
            min_cluster_size=3,  # Smaller for demo
            min_samples=2
        )
        print(f"   ✅ Clustering complete")
        
        # Show cluster summary
        cluster_summary = analyzer.get_cluster_summary()
        print("   📋 Cluster Summary:")
        for _, row in cluster_summary.iterrows():
            cluster_name = row['cluster_name']
            size = row['size']
            sample = row['sample_sentences'][0][:80] + "..." if row['sample_sentences'] else "No samples"
            print(f"      • {cluster_name}: {size} points - \"{sample}\"")
        print()
        
        # 4. Store results back to database
        print("💾 Step 4: Storing cluster labels back to database...")
        updated_count = analyzer.store_clusters_to_database()
        print(f"   ✅ Updated {updated_count} points in the database\n")
        
        # Save results for inspection
        print("💽 Saving results to file...")
        analyzer.save_results("demo_analysis_results.json")
        print("   ✅ Results saved to demo_analysis_results.json\n")
        
        # Final summary
        print("🎉 Demo Complete!")
        print(f"   📊 Analyzed {len(data)} company culture sentences")
        print(f"   🔍 Found {len(cluster_summary[cluster_summary['cluster_id'] >= 0])} meaningful clusters")
        print(f"   💾 Updated database with cluster information")
        print(f"   📁 Results saved for further analysis")
        
    except Exception as e:
        print(f"❌ Error during demo: {str(e)}")
        raise

if __name__ == "__main__":
    run_demo()
