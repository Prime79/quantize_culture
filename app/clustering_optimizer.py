#!/usr/bin/env python3
"""
Advanced clustering optimizer with automatic parameter tuning and benchmarking.

This module provides:
1. Automatic parameter grid search for UMAP + HDBSCAN
2. Quality scoring and benchmarking against previous results
3. Persistent benchmark storage and comparison
4. Comprehensive clustering workflow with optimization
"""

import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import umap
import hdbscan
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

class ClusteringOptimizer:
    """Advanced clustering optimizer with parameter tuning and benchmarking."""
    
    def __init__(self, benchmark_file: str = "clustering_benchmarks.json"):
        """Initialize the clustering optimizer."""
        self.benchmark_file = benchmark_file
        self.benchmarks = self._load_benchmarks()
        self.current_results = []
        self.best_params = None
        self.best_score = None
        
    def _load_benchmarks(self) -> Dict:
        """Load existing benchmarks from file."""
        if os.path.exists(self.benchmark_file):
            try:
                with open(self.benchmark_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️  Warning: Could not load benchmarks: {e}")
                return {"runs": [], "best_ever": None}
        return {"runs": [], "best_ever": None}
    
    def _save_benchmarks(self):
        """Save benchmarks to file."""
        try:
            with open(self.benchmark_file, 'w') as f:
                json.dump(self.benchmarks, f, indent=2, default=str)
        except Exception as e:
            print(f"⚠️  Warning: Could not save benchmarks: {e}")
    
    def _calculate_quality_score(self, 
                                n_clusters: int, 
                                noise_percentage: float, 
                                silhouette: float,
                                calinski_harabasz: float = 0,
                                davies_bouldin: float = float('inf')) -> float:
        """
        Calculate comprehensive quality score for clustering.
        
        Args:
            n_clusters: Number of clusters found
            noise_percentage: Percentage of noise points
            silhouette: Silhouette score
            calinski_harabasz: Calinski-Harabasz score
            davies_bouldin: Davies-Bouldin score (lower is better)
            
        Returns:
            Quality score (higher is better)
        """
        # Normalize components
        cluster_score = min(n_clusters / 10, 10)  # Favor moderate number of clusters
        noise_score = max(0, (100 - noise_percentage) / 10)  # Penalize high noise
        silhouette_score_norm = max(0, silhouette * 10)  # Amplify silhouette
        
        # Optional advanced metrics (if available)
        ch_score = min(calinski_harabasz / 1000, 10) if calinski_harabasz > 0 else 0
        db_score = max(0, 10 - davies_bouldin) if davies_bouldin != float('inf') else 0
        
        # Weighted combination
        quality_score = (
            cluster_score * 0.2 +
            noise_score * 0.4 +
            silhouette_score_norm * 0.3 +
            ch_score * 0.05 +
            db_score * 0.05
        )
        
        return quality_score
    
    def _get_parameter_grid(self) -> List[Dict]:
        """Define parameter grid for optimization."""
        return [
            # Previous best (baseline)
            {"name": "Baseline", "umap_neighbors": 15, "umap_min_dist": 0.1, 
             "hdbscan_min_cluster": 5, "hdbscan_min_samples": 3},
            
            # High-resolution clustering
            {"name": "High_Resolution", "umap_neighbors": 8, "umap_min_dist": 0.01, 
             "hdbscan_min_cluster": 2, "hdbscan_min_samples": 1},
            
            # Aggressive clustering
            {"name": "Aggressive", "umap_neighbors": 5, "umap_min_dist": 0.0, 
             "hdbscan_min_cluster": 3, "hdbscan_min_samples": 2},
            
            # Conservative clustering
            {"name": "Conservative", "umap_neighbors": 20, "umap_min_dist": 0.2, 
             "hdbscan_min_cluster": 8, "hdbscan_min_samples": 4},
            
            # Balanced approaches
            {"name": "Balanced_Tight", "umap_neighbors": 12, "umap_min_dist": 0.05, 
             "hdbscan_min_cluster": 4, "hdbscan_min_samples": 2},
            
            {"name": "Balanced_Loose", "umap_neighbors": 25, "umap_min_dist": 0.1, 
             "hdbscan_min_cluster": 6, "hdbscan_min_samples": 3},
            
            # Large neighborhood approaches
            {"name": "Large_Neighborhood", "umap_neighbors": 30, "umap_min_dist": 0.05, 
             "hdbscan_min_cluster": 3, "hdbscan_min_samples": 2},
            
            # Ultra-fine clustering
            {"name": "Ultra_Fine", "umap_neighbors": 6, "umap_min_dist": 0.001, 
             "hdbscan_min_cluster": 2, "hdbscan_min_samples": 1},
             
            # Moderate clustering
            {"name": "Moderate", "umap_neighbors": 15, "umap_min_dist": 0.05, 
             "hdbscan_min_cluster": 5, "hdbscan_min_samples": 2},
        ]
    
    def _test_single_parameter_set(self, 
                                  embeddings: np.ndarray, 
                                  params: Dict) -> Optional[Dict]:
        """Test a single parameter combination."""
        try:
            print(f"--- Testing {params['name']} ---")
            print(f"UMAP: neighbors={params['umap_neighbors']}, min_dist={params['umap_min_dist']}")
            print(f"HDBSCAN: min_cluster_size={params['hdbscan_min_cluster']}, min_samples={params['hdbscan_min_samples']}")
            
            # UMAP reduction
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=params['umap_neighbors'],
                min_dist=params['umap_min_dist'],
                metric='cosine',
                random_state=42,
                n_jobs=1
            )
            reduced_data = reducer.fit_transform(embeddings)
            
            # HDBSCAN clustering
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=params['hdbscan_min_cluster'],
                min_samples=params['hdbscan_min_samples'],
                metric='euclidean'
            )
            cluster_labels = clusterer.fit_predict(reduced_data)
            
            # Calculate metrics
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            n_noise = sum(1 for x in cluster_labels if x == -1)
            noise_percentage = (n_noise / len(cluster_labels)) * 100
            
            # Calculate quality metrics
            if n_clusters > 1:
                # Silhouette score (only for non-noise points)
                non_noise_mask = cluster_labels != -1
                if sum(non_noise_mask) > 1:
                    silhouette = silhouette_score(
                        reduced_data[non_noise_mask], 
                        cluster_labels[non_noise_mask]
                    )
                    
                    # Additional metrics
                    calinski_harabasz = calinski_harabasz_score(
                        reduced_data[non_noise_mask], 
                        cluster_labels[non_noise_mask]
                    )
                    davies_bouldin = davies_bouldin_score(
                        reduced_data[non_noise_mask], 
                        cluster_labels[non_noise_mask]
                    )
                else:
                    silhouette = 0
                    calinski_harabasz = 0
                    davies_bouldin = float('inf')
            else:
                silhouette = 0
                calinski_harabasz = 0
                davies_bouldin = float('inf')
            
            # Calculate cluster sizes
            cluster_sizes = {}
            for label in cluster_labels:
                if label != -1:
                    cluster_sizes[label] = cluster_sizes.get(label, 0) + 1
            
            avg_cluster_size = np.mean(list(cluster_sizes.values())) if cluster_sizes else 0
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(
                n_clusters, noise_percentage, silhouette, calinski_harabasz, davies_bouldin
            )
            
            result = {
                'name': params['name'],
                'params': params,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'noise_percentage': noise_percentage,
                'silhouette_score': silhouette,
                'calinski_harabasz_score': calinski_harabasz,
                'davies_bouldin_score': davies_bouldin,
                'quality_score': quality_score,
                'avg_cluster_size': avg_cluster_size,
                'cluster_sizes': cluster_sizes,
                'labels': cluster_labels.tolist(),
                'reduced_data': reduced_data.tolist()
            }
            
            print(f"Results: {n_clusters} clusters, {n_noise} noise points ({noise_percentage:.1f}%)")
            print(f"Cluster sizes: avg={avg_cluster_size:.1f}, min={min(cluster_sizes.values()) if cluster_sizes else 0}, max={max(cluster_sizes.values()) if cluster_sizes else 0}")
            print(f"Silhouette score: {silhouette:.3f}")
            print(f"Quality score: {quality_score:.1f}")
            print()
            
            return result
            
        except Exception as e:
            print(f"❌ Error testing {params['name']}: {str(e)}")
            return None
    
    def optimize_clustering(self, embeddings: np.ndarray) -> Dict:
        """
        Run parameter optimization for clustering.
        
        Args:
            embeddings: Input embeddings to cluster
            
        Returns:
            Best clustering result with parameters and metrics
        """
        print("🔧 Starting Clustering Parameter Optimization")
        print("=" * 60)
        
        # Standardize embeddings
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
        
        # Test all parameter combinations
        parameter_grid = self._get_parameter_grid()
        results = []
        
        for i, params in enumerate(parameter_grid, 1):
            print(f"Testing parameter set {i}/{len(parameter_grid)}")
            result = self._test_single_parameter_set(embeddings_scaled, params)
            if result:
                results.append(result)
        
        # Find best result
        if not results:
            raise ValueError("No successful clustering results found")
        
        best_result = max(results, key=lambda x: x['quality_score'])
        
        # Save results
        self.current_results = results
        self.best_params = best_result['params']
        self.best_score = best_result['quality_score']
        
        # Update benchmarks
        self._update_benchmarks(best_result)
        
        return best_result
    
    def _update_benchmarks(self, result: Dict):
        """Update benchmark history with new result."""
        benchmark_entry = {
            "timestamp": datetime.now().isoformat(),
            "name": result['name'],
            "params": result['params'],
            "n_clusters": result['n_clusters'],
            "noise_percentage": result['noise_percentage'],
            "silhouette_score": result['silhouette_score'],
            "quality_score": result['quality_score'],
            "avg_cluster_size": result['avg_cluster_size']
        }
        
        # Add to history
        self.benchmarks["runs"].append(benchmark_entry)
        
        # Update best ever if this is better
        if (self.benchmarks["best_ever"] is None or 
            result['quality_score'] > self.benchmarks["best_ever"]["quality_score"]):
            self.benchmarks["best_ever"] = benchmark_entry
            print(f"🎉 NEW BEST EVER RESULT!")
        
        # Keep only last 50 runs to avoid file bloat
        if len(self.benchmarks["runs"]) > 50:
            self.benchmarks["runs"] = self.benchmarks["runs"][-50:]
        
        # Save to file
        self._save_benchmarks()
    
    def print_optimization_summary(self):
        """Print comprehensive optimization summary."""
        if not self.current_results:
            print("No optimization results to display")
            return
        
        print("\n📊 CLUSTERING OPTIMIZATION SUMMARY")
        print("=" * 80)
        
        # Sort results by quality score
        sorted_results = sorted(self.current_results, key=lambda x: x['quality_score'], reverse=True)
        
        print(f"{'Method':<20} {'Clusters':<8} {'Noise %':<8} {'Silhouette':<11} {'Quality':<8} {'Avg Size':<8}")
        print("-" * 80)
        
        for result in sorted_results:
            print(f"{result['name']:<20} {result['n_clusters']:<8} {result['noise_percentage']:<8.1f} "
                  f"{result['silhouette_score']:<11.3f} {result['quality_score']:<8.1f} {result['avg_cluster_size']:<8.1f}")
        
        # Best result details
        best = sorted_results[0]
        print(f"\n🏆 BEST RESULT: {best['name']}")
        print(f"   Parameters: {best['params']}")
        print(f"   Quality Score: {best['quality_score']:.1f}")
        print(f"   Metrics: {best['n_clusters']} clusters, {best['noise_percentage']:.1f}% noise")
        print(f"   Silhouette: {best['silhouette_score']:.3f}")
        
        # Benchmark comparison
        if self.benchmarks["best_ever"]:
            best_ever = self.benchmarks["best_ever"]
            print(f"\n📈 BENCHMARK COMPARISON:")
            print(f"   Current Best: {best['quality_score']:.1f}")
            print(f"   Historical Best: {best_ever['quality_score']:.1f} ({best_ever['name']})")
            
            improvement = best['quality_score'] - best_ever['quality_score']
            if improvement > 0:
                print(f"   🎉 IMPROVEMENT: +{improvement:.1f} points!")
            elif improvement < -5:
                print(f"   ⚠️  REGRESSION: {improvement:.1f} points")
            else:
                print(f"   ✅ STABLE: {improvement:.1f} points difference")
    
    def create_optimization_visualization(self, save_path: str = "optimization_results.png"):
        """Create visualization of optimization results."""
        if not self.current_results:
            print("No results to visualize")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Clustering Optimization Results', fontsize=16)
        
        # Extract data for plotting
        names = [r['name'] for r in self.current_results]
        quality_scores = [r['quality_score'] for r in self.current_results]
        n_clusters = [r['n_clusters'] for r in self.current_results]
        noise_percentages = [r['noise_percentage'] for r in self.current_results]
        silhouette_scores = [r['silhouette_score'] for r in self.current_results]
        
        # Quality scores
        axes[0, 0].bar(range(len(names)), quality_scores)
        axes[0, 0].set_title('Quality Scores by Method')
        axes[0, 0].set_xlabel('Method')
        axes[0, 0].set_ylabel('Quality Score')
        axes[0, 0].set_xticks(range(len(names)))
        axes[0, 0].set_xticklabels(names, rotation=45, ha='right')
        
        # Clusters vs Noise
        axes[0, 1].scatter(n_clusters, noise_percentages, s=100, alpha=0.7)
        for i, name in enumerate(names):
            axes[0, 1].annotate(name, (n_clusters[i], noise_percentages[i]), 
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
        axes[0, 1].set_title('Clusters vs Noise Percentage')
        axes[0, 1].set_xlabel('Number of Clusters')
        axes[0, 1].set_ylabel('Noise Percentage')
        
        # Quality vs Silhouette
        axes[1, 0].scatter(silhouette_scores, quality_scores, s=100, alpha=0.7)
        for i, name in enumerate(names):
            axes[1, 0].annotate(name, (silhouette_scores[i], quality_scores[i]), 
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
        axes[1, 0].set_title('Quality Score vs Silhouette Score')
        axes[1, 0].set_xlabel('Silhouette Score')
        axes[1, 0].set_ylabel('Quality Score')
        
        # Historical benchmark trend
        axes[1, 1].set_title('Historical Quality Trend')
        if len(self.benchmarks["runs"]) > 1:
            timestamps = [r["timestamp"] for r in self.benchmarks["runs"][-20:]]  # Last 20 runs
            historical_scores = [r["quality_score"] for r in self.benchmarks["runs"][-20:]]
            axes[1, 1].plot(range(len(historical_scores)), historical_scores, 'o-')
            axes[1, 1].set_xlabel('Run Number (Recent)')
            axes[1, 1].set_ylabel('Quality Score')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'Not enough historical data', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Optimization visualization saved to {save_path}")


class EnhancedDataExtractorAnalyzer:
    """Enhanced analyzer with integrated clustering optimization."""
    
    def __init__(self, collection_name: str = "company_culture_embeddings"):
        """Initialize the enhanced analyzer."""
        # Import here to avoid circular imports
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from app.extract import DataExtractorAnalyzer
        
        self.base_analyzer = DataExtractorAnalyzer(collection_name)
        self.optimizer = ClusteringOptimizer()
        self.optimization_results = None
        
    def extract_data(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Extract data using base analyzer."""
        return self.base_analyzer.extract_data(limit=limit)
    
    def optimize_and_cluster(self, 
                           limit: Optional[int] = None,
                           save_visualization: bool = True) -> Dict:
        """
        Run complete optimization and clustering workflow.
        
        Args:
            limit: Maximum number of points to process
            save_visualization: Whether to save optimization plots
            
        Returns:
            Dictionary with optimization results and applied clustering
        """
        print("🚀 Starting Enhanced Clustering Workflow with Optimization")
        print("=" * 70)
        
        # Step 1: Extract data
        print("📊 Extracting data...")
        data = self.extract_data(limit=limit)
        embeddings = self.base_analyzer.embeddings
        print(f"   ✅ Loaded {len(embeddings)} embeddings")
        
        # Step 2: Optimize parameters
        print("\n🔧 Optimizing clustering parameters...")
        best_result = self.optimizer.optimize_clustering(embeddings)
        self.optimization_results = best_result
        
        # Step 3: Print optimization summary
        self.optimizer.print_optimization_summary()
        
        # Step 4: Apply best clustering to base analyzer
        print(f"\n⚙️  Applying best parameters to data...")
        best_params = best_result['params']
        
        # Apply UMAP reduction
        self.base_analyzer.reduce_dimensions(
            n_components=2,
            n_neighbors=best_params['umap_neighbors'],
            min_dist=best_params['umap_min_dist']
        )
        
        # Apply HDBSCAN clustering
        self.base_analyzer.cluster_data(
            min_cluster_size=best_params['hdbscan_min_cluster'],
            min_samples=best_params['hdbscan_min_samples']
        )
        
        # Step 5: Store results to database
        print("💾 Storing optimized clusters to database...")
        self.base_analyzer.store_clusters_to_database()
        
        # Step 6: Create visualizations
        if save_visualization:
            print("📊 Creating visualizations...")
            self.optimizer.create_optimization_visualization("optimization_results.png")
            
            self.base_analyzer.plot_clusters(
                figsize=(14, 10),
                save_path="optimized_clustering.png",
                title=f"Optimized Clustering ({best_result['name']})",
                alpha=0.8
            )
        
        # Step 7: Get final summary
        summary = self.base_analyzer.get_cluster_summary()
        
        print(f"\n✅ OPTIMIZATION WORKFLOW COMPLETE!")
        print(f"   Applied Method: {best_result['name']}")
        print(f"   Quality Score: {best_result['quality_score']:.1f}")
        print(f"   Final Clusters: {best_result['n_clusters']}")
        print(f"   Noise Percentage: {best_result['noise_percentage']:.1f}%")
        print(f"   Silhouette Score: {best_result['silhouette_score']:.3f}")
        
        return {
            'optimization_results': self.optimization_results,
            'applied_params': best_params,
            'cluster_summary': summary,
            'quality_metrics': {
                'quality_score': best_result['quality_score'],
                'n_clusters': best_result['n_clusters'],
                'noise_percentage': best_result['noise_percentage'],
                'silhouette_score': best_result['silhouette_score']
            }
        }
    
    def get_benchmark_history(self) -> Dict:
        """Get historical benchmark data."""
        return self.optimizer.benchmarks
    
    def get_current_optimization_results(self) -> Optional[Dict]:
        """Get current optimization results."""
        return self.optimization_results
