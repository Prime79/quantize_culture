#!/usr/bin/env python3
"""
Demonstration script showing the clustering optimization and benchmarking system.

This script demonstrates:
1. How parameter optimization works
2. Quality scoring and benchmarking
3. Comparison with previous results
4. Historical tracking of performance
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
from app.clustering_optimizer import EnhancedDataExtractorAnalyzer, ClusteringOptimizer

def simulate_poor_clustering():
    """Simulate poor clustering parameters to show benchmark comparison."""
    print("🔧 Simulating poor clustering parameters...")
    
    # Create a custom parameter set with poor performance
    poor_params = [
        # Extremely conservative (should produce lots of noise)
        {"name": "Poor_Conservative", "umap_neighbors": 50, "umap_min_dist": 0.5, 
         "hdbscan_min_cluster": 20, "hdbscan_min_samples": 10},
        
        # Too aggressive (should produce tiny clusters)
        {"name": "Poor_Aggressive", "umap_neighbors": 3, "umap_min_dist": 0.9, 
         "hdbscan_min_cluster": 50, "hdbscan_min_samples": 25},
    ]
    
    analyzer = EnhancedDataExtractorAnalyzer()
    
    # Extract data
    analyzer.extract_data(limit=200)
    embeddings = analyzer.base_analyzer.embeddings
    
    # Test poor parameters
    optimizer = ClusteringOptimizer()
    
    for params in poor_params:
        result = optimizer._test_single_parameter_set(embeddings, params)
        if result:
            print(f"Poor result: {result['name']} - Quality: {result['quality_score']:.1f}")
            
            # Manually add to benchmarks to show comparison
            benchmark_entry = {
                "timestamp": "2025-06-18T23:00:00.000000",
                "name": result['name'],
                "params": result['params'],
                "n_clusters": result['n_clusters'],
                "noise_percentage": result['noise_percentage'],
                "silhouette_score": result['silhouette_score'],
                "quality_score": result['quality_score'],
                "avg_cluster_size": result['avg_cluster_size']
            }
            optimizer.benchmarks["runs"].append(benchmark_entry)
    
    optimizer._save_benchmarks()
    print(f"   Added {len(poor_params)} poor results to benchmark history")

def demonstrate_optimization():
    """Demonstrate the full optimization workflow."""
    print("\n🚀 Demonstrating Full Optimization Workflow")
    print("=" * 60)
    
    # Run optimized clustering
    analyzer = EnhancedDataExtractorAnalyzer()
    results = analyzer.optimize_and_cluster(limit=200, save_visualization=True)
    
    # Show detailed results
    print(f"\n📊 DETAILED RESULTS:")
    print(f"   Method: {results['applied_params']['name']}")
    print(f"   Parameters: {results['applied_params']}")
    print(f"   Quality Score: {results['quality_metrics']['quality_score']:.1f}")
    print(f"   Clusters: {results['quality_metrics']['n_clusters']}")
    print(f"   Noise: {results['quality_metrics']['noise_percentage']:.1f}%")
    print(f"   Silhouette: {results['quality_metrics']['silhouette_score']:.3f}")
    
    # Show benchmark history
    history = analyzer.get_benchmark_history()
    print(f"\n📈 BENCHMARK HISTORY:")
    print(f"   Total runs: {len(history['runs'])}")
    print(f"   Best ever quality: {history['best_ever']['quality_score']:.1f}")
    print(f"   Best ever method: {history['best_ever']['name']}")
    
    if len(history['runs']) > 1:
        print(f"\n📊 RECENT PERFORMANCE TREND:")
        recent_scores = [r['quality_score'] for r in history['runs'][-5:]]
        recent_names = [r['name'] for r in history['runs'][-5:]]
        
        for i, (name, score) in enumerate(zip(recent_names, recent_scores)):
            print(f"   {i+1}. {name}: {score:.1f}")
    
    return results

def show_quality_scoring_details():
    """Show how quality scoring works with different scenarios."""
    print("\n🔍 Quality Scoring System Details")
    print("=" * 40)
    
    optimizer = ClusteringOptimizer()
    
    # Test different scenarios
    scenarios = [
        {"name": "Perfect", "n_clusters": 20, "noise_pct": 0, "silhouette": 0.8},
        {"name": "Good", "n_clusters": 15, "noise_pct": 5, "silhouette": 0.6},
        {"name": "Average", "n_clusters": 10, "noise_pct": 15, "silhouette": 0.4},
        {"name": "Poor", "n_clusters": 3, "noise_pct": 60, "silhouette": 0.2},
        {"name": "Terrible", "n_clusters": 1, "noise_pct": 90, "silhouette": 0.1},
    ]
    
    print(f"{'Scenario':<10} {'Clusters':<8} {'Noise %':<8} {'Silhouette':<11} {'Quality':<8}")
    print("-" * 50)
    
    for scenario in scenarios:
        quality = optimizer._calculate_quality_score(
            scenario['n_clusters'], 
            scenario['noise_pct'], 
            scenario['silhouette']
        )
        print(f"{scenario['name']:<10} {scenario['n_clusters']:<8} {scenario['noise_pct']:<8} "
              f"{scenario['silhouette']:<11.1f} {quality:<8.1f}")

def main():
    """Run the complete demonstration."""
    print("🎯 Clustering Optimization & Benchmarking Demonstration")
    print("=" * 70)
    
    # Step 1: Show quality scoring
    show_quality_scoring_details()
    
    # Step 2: Simulate poor clustering for comparison
    simulate_poor_clustering()
    
    # Step 3: Run optimization
    results = demonstrate_optimization()
    
    # Step 4: Show final summary
    print(f"\n🎉 DEMONSTRATION COMPLETE!")
    print(f"   Final Quality Score: {results['quality_metrics']['quality_score']:.1f}")
    print(f"   Benchmark Status: {'NEW RECORD!' if results['quality_metrics']['quality_score'] > 7.0 else 'STABLE'}")
    
    # Show generated files
    print(f"\n📁 Generated Files:")
    for filename in ["optimization_results.png", "optimized_clustering.png", 
                    "clustering_benchmarks.json", "out_01.json"]:
        if os.path.exists(filename):
            print(f"   ✅ {filename}")
        else:
            print(f"   ❌ {filename}")

if __name__ == "__main__":
    main()
