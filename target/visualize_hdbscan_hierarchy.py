#!/usr/bin/env python3
"""
Visualize HDBSCAN Cluster Hierarchy

This script runs HDBSCAN on the 3D UMAP data and visualizes the
resulting cluster hierarchy using a dendrogram to show how clusters
merge and split at different distance scales.
"""

import pandas as pd
import numpy as np
import hdbscan
import matplotlib.pyplot as plt
import seaborn as sns
from qdrant_client import QdrantClient
import umap
import warnings

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid')

def fetch_data_from_qdrant():
    """Fetch original embeddings and metadata from Qdrant."""
    print("📊 Fetching data from Qdrant collection 'target_collection'...")
    client = QdrantClient(path="qdrant_data")
    
    points, _ = client.scroll(
        collection_name="target_collection",
        limit=10000,  # Fetch all points
        with_payload=True,
        with_vectors=True
    )
    
    print(f"✅ Fetched {len(points)} data points.")
    return points

def perform_3d_umap(points):
    """Perform 3D UMAP reduction on high-dimensional embeddings."""
    print("🗺️  Performing 3D UMAP reduction...")
    
    vectors = [point.vector for point in points]
    
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=15,
        min_dist=0.1,
        metric='cosine',
        random_state=42
    )
    
    umap_3d = reducer.fit_transform(vectors)
    print("✅ 3D UMAP reduction complete.")
    return umap_3d

def visualize_hierarchy(umap_3d):
    """Run HDBSCAN and visualize the cluster hierarchy."""
    print("🧠 Running HDBSCAN to generate cluster hierarchy...")
    
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=15,
        min_samples=5,
        gen_min_span_tree=True  # Required for plotting the hierarchy
    )
    
    clusterer.fit(umap_3d)
    
    num_clusters = len(set(clusterer.labels_)) - (1 if -1 in clusterer.labels_ else 0)
    num_noise = np.sum(clusterer.labels_ == -1)
    
    print(f"🎯 HDBSCAN found {num_clusters} stable clusters and {num_noise} noise points.")
    
    print("📈 Plotting the condensed cluster hierarchy (dendrogram)...")
    
    plt.figure(figsize=(20, 10))
    
    # Plot the condensed tree
    clusterer.condensed_tree_.plot(
        select_clusters=True,
        selection_palette=sns.color_palette('deep', num_clusters)
    )
    
    plt.title('HDBSCAN Condensed Cluster Hierarchy', fontsize=16, fontweight='bold')
    plt.xlabel('Sample Index')
    plt.ylabel('Lambda (Distance)')
    
    hierarchy_plot_path = 'hdbscan_cluster_hierarchy.png'
    plt.savefig(hierarchy_plot_path, dpi=300, bbox_inches='tight')
    
    print(f"💾 Hierarchy plot saved as '{hierarchy_plot_path}'")
    
    print("\n💡 HOW TO READ THE PLOT:")
    print("   • The Y-axis (Lambda) represents the distance at which points are merged.")
    print("   • Vertical lines show individual data points or small clusters.")
    print("   • Horizontal lines show where clusters are merged.")
    print("   • Thicker, more colorful branches represent the stable clusters that HDBSCAN selected.")
    print("   • You can see how the two main clusters are formed from smaller sub-clusters.")

def main():
    """Main function to run the hierarchy visualization."""
    points = fetch_data_from_qdrant()
    umap_3d = perform_3d_umap(points)
    visualize_hierarchy(umap_3d)

if __name__ == "__main__":
    main()
