#!/usr/bin/env python3
"""
Script to test and improve clustering quality by trying different parameter combinations.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

from app.extract import DataExtractorAnalyzer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import umap
import hdbscan

def test_clustering_parameters():
    """Test different parameter combinations to improve clustering quality"""
    
    print("🔧 Testing Different Clustering Parameters\n")
    
    analyzer = DataExtractorAnalyzer()
    
    # Extract data once
    print("📊 Extracting data...")
    analyzer.extract_data(limit=200)
    embeddings = analyzer.embeddings
    print(f"   ✅ Loaded {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
    
    # Standardize embeddings
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    # Test different parameter combinations
    parameter_sets = [
        # Current (poor performance)
        {"name": "Current", "umap_neighbors": 15, "umap_min_dist": 0.1, "hdbscan_min_cluster": 5, "hdbscan_min_samples": 3},
        
        # More aggressive clustering - smaller neighborhoods
        {"name": "Aggressive", "umap_neighbors": 5, "umap_min_dist": 0.0, "hdbscan_min_cluster": 3, "hdbscan_min_samples": 2},
        
        # Larger neighborhoods, tighter clusters
        {"name": "Large_Neighborhood", "umap_neighbors": 30, "umap_min_dist": 0.05, "hdbscan_min_cluster": 3, "hdbscan_min_samples": 2},
        
        # Very tight clusters
        {"name": "Very_Tight", "umap_neighbors": 8, "umap_min_dist": 0.01, "hdbscan_min_cluster": 2, "hdbscan_min_samples": 1},
        
        # Conservative approach
        {"name": "Conservative", "umap_neighbors": 20, "umap_min_dist": 0.2, "hdbscan_min_cluster": 8, "hdbscan_min_samples": 4},
        
        # Balanced approach
        {"name": "Balanced", "umap_neighbors": 12, "umap_min_dist": 0.05, "hdbscan_min_cluster": 4, "hdbscan_min_samples": 2},
        
        # High resolution
        {"name": "High_Resolution", "umap_neighbors": 25, "umap_min_dist": 0.0, "hdbscan_min_cluster": 2, "hdbscan_min_samples": 1},
    ]
    
    results = []
    
    for i, params in enumerate(parameter_sets):
        print(f"\n--- Testing {params['name']} ({i+1}/{len(parameter_sets)}) ---")
        print(f"UMAP: neighbors={params['umap_neighbors']}, min_dist={params['umap_min_dist']}")
        print(f"HDBSCAN: min_cluster_size={params['hdbscan_min_cluster']}, min_samples={params['hdbscan_min_samples']}")
        
        try:
            # Apply UMAP
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=params['umap_neighbors'],
                min_dist=params['umap_min_dist'],
                metric='cosine',
                random_state=42
            )
            reduced_data = reducer.fit_transform(embeddings_scaled)
            
            # Apply HDBSCAN
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=params['hdbscan_min_cluster'],
                min_samples=params['hdbscan_min_samples'],
                metric='euclidean'
            )
            cluster_labels = clusterer.fit_predict(reduced_data)
            
            # Analyze results
            unique_clusters = set(cluster_labels)
            n_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
            noise_count = sum(1 for label in cluster_labels if label == -1)
            noise_percentage = (noise_count / len(cluster_labels)) * 100
            
            # Calculate cluster sizes
            cluster_sizes = {}
            for label in cluster_labels:
                if label != -1:
                    cluster_sizes[label] = cluster_sizes.get(label, 0) + 1
            
            # Calculate silhouette score if possible
            silhouette = -1
            if n_clusters > 1 and noise_count < len(cluster_labels):
                try:
                    valid_points = cluster_labels != -1
                    if np.sum(valid_points) > 1:
                        silhouette = silhouette_score(reduced_data[valid_points], 
                                                    cluster_labels[valid_points])
                except:
                    pass
            
            print(f"Results: {n_clusters} clusters, {noise_count} noise points ({noise_percentage:.1f}%)")
            
            if cluster_sizes:
                avg_cluster_size = np.mean(list(cluster_sizes.values()))
                min_cluster_size = min(cluster_sizes.values())
                max_cluster_size = max(cluster_sizes.values())
                print(f"Cluster sizes: avg={avg_cluster_size:.1f}, min={min_cluster_size}, max={max_cluster_size}")
                print(f"Silhouette score: {silhouette:.3f}")
            
            results.append({
                'name': params['name'],
                'params': params,
                'n_clusters': n_clusters,
                'noise_percentage': noise_percentage,
                'cluster_sizes': cluster_sizes,
                'labels': cluster_labels,
                'reduced_data': reduced_data,
                'silhouette_score': silhouette,
                'success': True
            })
            
        except Exception as e:
            print(f"   ❌ Failed: {str(e)}")
            results.append({
                'name': params['name'],
                'params': params,
                'success': False,
                'error': str(e)
            })
    
    return results, analyzer

def plot_parameter_comparison(results):
    """Plot comparison of different parameter sets"""
    
    successful_results = [r for r in results if r.get('success', False)]
    
    if len(successful_results) == 0:
        print("No successful clustering results to plot")
        return None
    
    n_plots = min(len(successful_results), 6)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, result in enumerate(successful_results[:n_plots]):
        ax = axes[i]
        reduced_data = result['reduced_data']
        cluster_labels = result['labels']
        
        # Plot clusters
        unique_labels = sorted(set(cluster_labels))
        colors = plt.cm.Set3(np.linspace(0, 1, max(len(unique_labels), 1)))
        
        for j, label in enumerate(unique_labels):
            mask = np.array(cluster_labels) == label
            if label == -1:
                ax.scatter(reduced_data[mask, 0], reduced_data[mask, 1], 
                          c='lightgray', alpha=0.3, s=20, label='Noise')
            else:
                ax.scatter(reduced_data[mask, 0], reduced_data[mask, 1], 
                          c=[colors[j % len(colors)]], alpha=0.7, s=30, 
                          label=f'Cluster {label}')
        
        noise_pct = result['noise_percentage']
        n_clusters = result['n_clusters']
        silhouette = result['silhouette_score']
        
        ax.set_title(f"{result['name']}\n{n_clusters} clusters, {noise_pct:.1f}% noise\nSilhouette: {silhouette:.3f}")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('UMAP Dimension 1')
        ax.set_ylabel('UMAP Dimension 2')
    
    # Hide unused subplots
    for i in range(n_plots, 6):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('parameter_comparison.png', dpi=300, bbox_inches='tight')
    print("   📁 Comparison plot saved to parameter_comparison.png")
    plt.close()
    
    return successful_results

def analyze_results(results):
    """Analyze and rank the clustering results"""
    
    successful_results = [r for r in results if r.get('success', False)]
    
    if not successful_results:
        print("❌ No successful clustering results to analyze")
        return None
    
    print(f"\n📊 CLUSTERING RESULTS ANALYSIS")
    print(f"{'='*80}")
    print(f"{'Method':<15} {'Clusters':<8} {'Noise %':<8} {'Silhouette':<12} {'Quality Score':<12}")
    print(f"{'-'*80}")
    
    # Calculate quality scores (lower noise + higher silhouette + reasonable cluster count)
    for result in successful_results:
        n_clusters = result['n_clusters']
        noise_pct = result['noise_percentage']
        silhouette = result['silhouette_score']
        
        # Quality score: prefer 3-12 clusters, low noise, high silhouette
        cluster_penalty = 0
        if n_clusters < 3:
            cluster_penalty = 20  # Too few clusters
        elif n_clusters > 15:
            cluster_penalty = 10  # Too many clusters
        
        silhouette_score = max(0, silhouette) * 100  # Convert to 0-100 scale
        quality_score = max(0, 100 - noise_pct - cluster_penalty + silhouette_score)
        
        result['quality_score'] = quality_score
        
        print(f"{result['name']:<15} {n_clusters:<8} {noise_pct:<8.1f} {silhouette:<12.3f} {quality_score:<12.1f}")
    
    # Sort by quality score
    successful_results.sort(key=lambda x: x['quality_score'], reverse=True)
    
    print(f"\n🏆 BEST CLUSTERING APPROACH:")
    best = successful_results[0]
    print(f"   Method: {best['name']}")
    print(f"   Parameters: {best['params']}")
    print(f"   Results: {best['n_clusters']} clusters, {best['noise_percentage']:.1f}% noise")
    print(f"   Silhouette score: {best['silhouette_score']:.3f}")
    print(f"   Quality score: {best['quality_score']:.1f}")
    
    # Show cluster distribution
    if best['cluster_sizes']:
        print(f"\n   Cluster sizes:")
        for cluster_id, size in sorted(best['cluster_sizes'].items()):
            print(f"      Cluster {cluster_id}: {size} sentences")
    
    return best

def apply_best_clustering(best_result, analyzer):
    """Apply the best clustering parameters to all data and save results"""
    
    if not best_result:
        print("❌ No best result to apply")
        return
    
    print(f"\n🔧 Applying best clustering parameters to all data...")
    
    # Get the best parameters
    params = best_result['params']
    
    # Apply to analyzer
    print("   🎯 Reducing dimensions...")
    analyzer.reduce_dimensions(
        n_components=2,
        n_neighbors=params['umap_neighbors'],
        min_dist=params['umap_min_dist']
    )
    
    print("   🔍 Clustering...")
    analyzer.cluster_data(
        min_cluster_size=params['hdbscan_min_cluster'],
        min_samples=params['hdbscan_min_samples']
    )
    
    # Store results to database
    print("   💾 Storing improved clusters to database...")
    analyzer.store_clusters_to_database()
    
    # Create visualization
    print("   📈 Creating improved clustering plot...")
    analyzer.plot_clusters(
        figsize=(14, 10),
        save_path="improved_clustering.png",
        title=f"Improved Company Culture Clustering ({best_result['name']})",
        alpha=0.8
    )
    
    # Show final summary
    summary = analyzer.get_cluster_summary()
    print(f"\n✅ IMPROVED CLUSTERING APPLIED:")
    print(f"   Total sentences: {len(analyzer.data)}")
    print(f"   Clusters found: {len(summary[summary['cluster_id'] >= 0])}")
    print(f"   Noise points: {sum(summary['cluster_id'] == -1)}")
    print(f"   Noise percentage: {sum(summary['cluster_id'] == -1)/len(analyzer.data)*100:.1f}%")
    
    return analyzer

def main():
    """Run the complete clustering improvement process"""
    print("🚀 Starting Clustering Quality Improvement Process\n")
    
    try:
        # Step 1: Test different parameters
        results, analyzer = test_clustering_parameters()
        
        # Step 2: Plot comparisons
        print(f"\n📈 Creating comparison plots...")
        successful_results = plot_parameter_comparison(results)
        
        # Step 3: Analyze results
        best_result = analyze_results(results)
        
        # Step 4: Apply best clustering
        if best_result:
            improved_analyzer = apply_best_clustering(best_result, analyzer)
            
            print(f"\n🎉 Clustering improvement complete!")
            print(f"📁 Generated files:")
            print(f"   • parameter_comparison.png - Parameter comparison plots")
            print(f"   • improved_clustering.png - Final improved clustering")
            
            # Update the output JSON with improved clusters
            print(f"\n🔄 Updating out_01.json with improved clusters...")
            os.system("python create_enhanced_output_simple.py")
            
        else:
            print(f"\n❌ Could not find improved clustering parameters")
        
    except Exception as e:
        print(f"❌ Error during clustering improvement: {str(e)}")
        raise

if __name__ == "__main__":
    main()
