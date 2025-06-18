#!/usr/bin/env python3
"""
Extract and analyze data from Qdrant vector database.

This script provides four main functionalities:
1. Extract data from the database (all or specific)
2. Perform UMAP dimensionality reduction (2D or 3D)
3. Run HDBSCAN clustering on the reduced data
4. Store cluster labels back to the original embeddings in the database
"""

import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, Range, MatchValue
import umap
import hdbscan
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load environment variables
load_dotenv()

class DataExtractorAnalyzer:
    """Main class for extracting and analyzing vector data from Qdrant."""
    
    def __init__(self, collection_name: str = "company_culture_embeddings"):
        """Initialize the analyzer with Qdrant connection."""
        self.collection_name = collection_name
        self.client = QdrantClient(host="localhost", port=6333)
        self.data = None
        self.embeddings = None
        self.reduced_embeddings = None
        self.cluster_labels = None
        
    def extract_data(self, limit: Optional[int] = None, 
                    filter_conditions: Optional[Dict] = None) -> pd.DataFrame:
        """
        Extract data from the Qdrant database.
        
        Args:
            limit: Maximum number of points to extract (None for all)
            filter_conditions: Optional filter conditions for specific data extraction
            
        Returns:
            DataFrame with point IDs, payloads, and embeddings
        """
        print(f"Extracting data from collection '{self.collection_name}'...")
        
        # Build filter if provided
        query_filter = None
        if filter_conditions:
            # Example: {"company": "TechCorp", "rating_min": 3}
            conditions = []
            for key, value in filter_conditions.items():
                if key.endswith("_min"):
                    field_name = key.replace("_min", "")
                    conditions.append(FieldCondition(key=field_name, range=Range(gte=value)))
                elif key.endswith("_max"):
                    field_name = key.replace("_max", "")
                    conditions.append(FieldCondition(key=field_name, range=Range(lte=value)))
                else:
                    conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
            
            if conditions:
                query_filter = Filter(must=conditions)
        
        # Scroll through all points
        points = []
        offset = None
        batch_size = 100
        
        while True:
            result = self.client.scroll(
                collection_name=self.collection_name,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=True,
                scroll_filter=query_filter
            )
            
            if not result[0]:  # No more points
                break
                
            points.extend(result[0])
            offset = result[1]  # Next offset
            
            if limit and len(points) >= limit:
                points = points[:limit]
                break
        
        print(f"Extracted {len(points)} points from the database")
        
        # Convert to DataFrame
        data_rows = []
        embeddings = []
        
        for point in points:
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
            embeddings.append(point.vector)
        
        self.data = pd.DataFrame(data_rows)
        self.embeddings = np.array(embeddings)
        
        print(f"Data shape: {self.data.shape}")
        print(f"Embeddings shape: {self.embeddings.shape}")
        
        return self.data
    
    def reduce_dimensions(self, n_components: int = 2, 
                         random_state: int = 42,
                         n_neighbors: int = 15,
                         min_dist: float = 0.1,
                         metric: str = 'cosine') -> np.ndarray:
        """
        Perform UMAP dimensionality reduction on the embeddings.
        
        Args:
            n_components: Number of dimensions to reduce to (2 or 3)
            random_state: Random state for reproducibility
            n_neighbors: Number of neighbors for UMAP
            min_dist: Minimum distance for UMAP
            metric: Distance metric to use
            
        Returns:
            Reduced embeddings array
        """
        if self.embeddings is None:
            raise ValueError("No embeddings found. Please extract data first.")
        
        print(f"Reducing dimensions from {self.embeddings.shape[1]} to {n_components}...")
        
        # Standardize embeddings (optional but often helpful)
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(self.embeddings)
        
        # Initialize UMAP
        reducer = umap.UMAP(
            n_components=n_components,
            random_state=random_state,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            verbose=True
        )
        
        # Fit and transform
        self.reduced_embeddings = reducer.fit_transform(embeddings_scaled)
        
        print(f"Reduced embeddings shape: {self.reduced_embeddings.shape}")
        
        # Add reduced dimensions to DataFrame
        for i in range(n_components):
            self.data[f'umap_{i+1}'] = self.reduced_embeddings[:, i]
        
        return self.reduced_embeddings
    
    def cluster_data(self, min_cluster_size: int = 5,
                    min_samples: Optional[int] = None,
                    cluster_selection_epsilon: float = 0.0,
                    metric: str = 'euclidean') -> np.ndarray:
        """
        Perform HDBSCAN clustering on the reduced embeddings.
        
        Args:
            min_cluster_size: Minimum size of clusters
            min_samples: Minimum samples in a cluster (defaults to min_cluster_size)
            cluster_selection_epsilon: Epsilon for cluster selection
            metric: Distance metric for clustering
            
        Returns:
            Cluster labels array
        """
        if self.reduced_embeddings is None:
            raise ValueError("No reduced embeddings found. Please reduce dimensions first.")
        
        print(f"Clustering data with HDBSCAN...")
        
        if min_samples is None:
            min_samples = min_cluster_size
        
        # Initialize HDBSCAN
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            metric=metric
        )
        
        # Fit and predict
        self.cluster_labels = clusterer.fit_predict(self.reduced_embeddings)
        
        # Get cluster statistics
        unique_labels = np.unique(self.cluster_labels)
        n_clusters = len(unique_labels[unique_labels >= 0])  # Exclude noise (-1)
        n_noise = np.sum(self.cluster_labels == -1)
        
        print(f"Found {n_clusters} clusters")
        print(f"Noise points: {n_noise}")
        print(f"Cluster sizes: {np.bincount(self.cluster_labels[self.cluster_labels >= 0])}")
        
        # Add cluster labels to DataFrame
        self.data['cluster'] = self.cluster_labels
        
        return self.cluster_labels
    
    def store_clusters_to_database(self) -> int:
        """
        Store cluster labels back to the original embeddings in the database.
        
        Returns:
            Number of points updated
        """
        if self.cluster_labels is None:
            raise ValueError("No cluster labels found. Please run clustering first.")
        
        print("Storing cluster labels back to the database...")
        
        updated_count = 0
        
        # Update each point with its cluster label
        for idx, (point_id, cluster_label) in enumerate(zip(self.data['point_id'], self.cluster_labels)):
            # Update the point's payload with cluster information
            self.client.set_payload(
                collection_name=self.collection_name,
                payload={
                    "cluster": int(cluster_label),
                    "cluster_name": f"cluster_{cluster_label}" if cluster_label >= 0 else "noise"
                },
                points=[point_id]
            )
            updated_count += 1
            
            if updated_count % 50 == 0:
                print(f"Updated {updated_count}/{len(self.data)} points...")
        
        print(f"Successfully updated {updated_count} points with cluster labels")
        return updated_count
    
    def save_results(self, filename: str = "analysis_results.json") -> None:
        """Save analysis results to a JSON file."""
        if self.data is None:
            raise ValueError("No data to save. Please extract data first.")
        
        results = {
            "metadata": {
                "total_points": len(self.data),
                "embedding_dimension": self.embeddings.shape[1] if self.embeddings is not None else None,
                "reduced_dimension": self.reduced_embeddings.shape[1] if self.reduced_embeddings is not None else None,
                "n_clusters": len(np.unique(self.cluster_labels[self.cluster_labels >= 0])) if self.cluster_labels is not None else None,
                "n_noise_points": np.sum(self.cluster_labels == -1) if self.cluster_labels is not None else None
            },
            "data": self.data.to_dict('records')
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Results saved to {filename}")
    
    def get_cluster_summary(self) -> pd.DataFrame:
        """Get a summary of clusters with sample sentences."""
        if self.cluster_labels is None:
            raise ValueError("No cluster labels found. Please run clustering first.")
        
        summaries = []
        
        for cluster_id in sorted(np.unique(self.cluster_labels)):
            cluster_data = self.data[self.data['cluster'] == cluster_id]
            
            # Get sample sentences
            sample_sentences = cluster_data['sentence'].head(3).tolist()
            
            summary = {
                'cluster_id': cluster_id,
                'cluster_name': f"cluster_{cluster_id}" if cluster_id >= 0 else "noise",
                'size': len(cluster_data),
                'categories': cluster_data['category'].value_counts().to_dict(),
                'sample_sentences': sample_sentences
            }
            summaries.append(summary)
        
        return pd.DataFrame(summaries)
    
    def plot_clusters(self, 
                     figsize: Tuple[int, int] = (12, 8),
                     save_path: Optional[str] = None,
                     show_labels: bool = True,
                     alpha: float = 0.7,
                     title: str = "UMAP Clustering Results") -> None:
        """
        Plot the UMAP clustering results.
        
        Args:
            figsize: Figure size tuple
            save_path: Path to save the plot
            show_labels: Whether to show cluster labels
            alpha: Point transparency
            title: Plot title
        """
        from .utils import plot_umap_clusters
        
        if self.reduced_embeddings is None or self.cluster_labels is None:
            raise ValueError("Must run dimensionality reduction and clustering first")
        
        plot_umap_clusters(
            self.data,
            self.reduced_embeddings,
            self.cluster_labels,
            figsize=figsize,
            save_path=save_path,
            show_labels=show_labels,
            alpha=alpha,
            title=title
        )
    
    def plot_cluster_comparison(self,
                               figsize: Tuple[int, int] = (16, 6),
                               save_path: Optional[str] = None) -> None:
        """
        Create a comparison plot of 2D vs 3D clustering.
        
        Args:
            figsize: Figure size tuple
            save_path: Path to save the plot
        """
        from .utils import plot_cluster_comparison
        
        plot_cluster_comparison(self, figsize=figsize, save_path=save_path)
    
    def create_all_plots(self,
                        save_2d: str = "umap_2d_clusters.png",
                        save_3d: str = "umap_3d_clusters.png", 
                        save_comparison: str = "umap_comparison.png",
                        save_summary: str = "cluster_summary.png") -> None:
        """
        Create all visualization plots.
        
        Args:
            save_2d: Path for 2D plot
            save_3d: Path for 3D plot  
            save_comparison: Path for comparison plot
            save_summary: Path for summary chart
        """
        from .utils import create_all_plots
        
        create_all_plots(self, save_2d, save_3d, save_comparison, save_summary)
    
    def run_optimized_clustering(self, limit: Optional[int] = None) -> Dict:
        """
        Run optimized clustering workflow with automatic parameter tuning.
        
        This method uses the enhanced clustering optimizer to automatically
        find the best parameters and benchmark against historical results.
        
        Args:
            limit: Maximum number of points to process
            
        Returns:
            Dictionary with optimization results and quality metrics
        """
        try:
            from .clustering_optimizer import EnhancedDataExtractorAnalyzer
            
            # Use enhanced analyzer for optimization
            enhanced_analyzer = EnhancedDataExtractorAnalyzer(self.collection_name)
            results = enhanced_analyzer.optimize_and_cluster(limit=limit)
            
            # Copy results back to this analyzer
            self.data = enhanced_analyzer.base_analyzer.data
            self.embeddings = enhanced_analyzer.base_analyzer.embeddings
            self.reduced_embeddings = enhanced_analyzer.base_analyzer.reduced_embeddings
            self.cluster_labels = enhanced_analyzer.base_analyzer.cluster_labels
            
            return results
            
        except ImportError as e:
            print(f"⚠️  Enhanced clustering not available: {e}")
            print("   Falling back to basic clustering...")
            
            # Fallback to basic clustering
            self.extract_data(limit=limit)
            self.reduce_dimensions()
            self.cluster_data()
            self.store_clusters_to_database()
            
            return {
                'fallback': True,
                'message': 'Used basic clustering due to import error'
            }

def main():
    """Main function demonstrating the full pipeline."""
    print("=== Company Culture Data Analysis Pipeline ===\n")
    
    # Initialize analyzer
    analyzer = DataExtractorAnalyzer()
    
    try:
        # 1. Extract data
        print("Step 1: Extracting data from database...")
        data = analyzer.extract_data(limit=None)  # Extract all data
        
        # 2. Reduce dimensions with UMAP
        print("\nStep 2: Reducing dimensions with UMAP...")
        reduced_embeddings = analyzer.reduce_dimensions(
            n_components=2,  # Change to 3 for 3D
            n_neighbors=15,
            min_dist=0.1
        )
        
        # 3. Cluster with HDBSCAN
        print("\nStep 3: Clustering with HDBSCAN...")
        cluster_labels = analyzer.cluster_data(
            min_cluster_size=5,
            min_samples=3
        )
        
        # 4. Store results back to database
        print("\nStep 4: Storing cluster labels to database...")
        updated_count = analyzer.store_clusters_to_database()
        
        # Display results
        print("\n=== Analysis Results ===")
        cluster_summary = analyzer.get_cluster_summary()
        print(cluster_summary)
        
        # Save results
        analyzer.save_results("culture_analysis_results.json")
        
        print(f"\n✅ Analysis complete! Updated {updated_count} points in the database.")
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        raise


if __name__ == "__main__":
    main()
