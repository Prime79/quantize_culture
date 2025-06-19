#!/usr/bin/env python3
"""
Utility functions for quantize_culture project.
Contains plotting and visualization functions for UMAP clustering results.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List
import seaborn as sns
import datetime
import os

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def plot_umap_clusters(data: pd.DataFrame, 
                      reduced_embeddings: np.ndarray,
                      cluster_labels: np.ndarray,
                      figsize: Tuple[int, int] = (12, 8),
                      save_path: Optional[str] = None,
                      show_labels: bool = True,
                      alpha: float = 0.7,
                      title: str = "UMAP Clustering Results") -> None:
    """
    Plot UMAP clustering results in 2D or 3D.
    
    Args:
        data: DataFrame with original data
        reduced_embeddings: UMAP reduced embeddings
        cluster_labels: Cluster labels from HDBSCAN
        figsize: Figure size tuple
        save_path: Path to save the plot
        show_labels: Whether to show cluster labels
        alpha: Point transparency
        title: Plot title
    """
    n_dimensions = reduced_embeddings.shape[1]
    
    # Get unique clusters and create color palette
    unique_clusters = sorted(set(cluster_labels))
    n_clusters = len([c for c in unique_clusters if c >= 0])
    colors = plt.cm.Set3(np.linspace(0, 1, max(n_clusters, 1)))
    
    if n_dimensions == 2:
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot each cluster
        for i, cluster_id in enumerate(unique_clusters):
            mask = cluster_labels == cluster_id
            points = reduced_embeddings[mask]
            
            if cluster_id == -1:
                # Noise points
                ax.scatter(points[:, 0], points[:, 1], 
                          c='lightgray', alpha=alpha*0.5, s=30, 
                          label=f'Noise ({np.sum(mask)} points)', 
                          edgecolors='gray', linewidth=0.5)
            else:
                ax.scatter(points[:, 0], points[:, 1], 
                          c=[colors[i % len(colors)]], alpha=alpha, s=60, 
                          label=f'Cluster {cluster_id} ({np.sum(mask)} points)',
                          edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel('UMAP Dimension 1', fontsize=12)
        ax.set_ylabel('UMAP Dimension 2', fontsize=12)
        ax.set_title(f'{title}\n({len(data)} points, {n_clusters} clusters)', fontsize=14, fontweight='bold')
        
        if show_labels:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('white')
        
    elif n_dimensions == 3:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot each cluster
        for i, cluster_id in enumerate(unique_clusters):
            mask = cluster_labels == cluster_id
            points = reduced_embeddings[mask]
            
            if cluster_id == -1:
                # Noise points
                ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                          c='lightgray', alpha=alpha*0.5, s=30, 
                          label=f'Noise ({np.sum(mask)} points)')
            else:
                ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                          c=[colors[i % len(colors)]], alpha=alpha, s=60, 
                          label=f'Cluster {cluster_id} ({np.sum(mask)} points)')
        
        ax.set_xlabel('UMAP Dimension 1', fontsize=12)
        ax.set_ylabel('UMAP Dimension 2', fontsize=12)
        ax.set_zlabel('UMAP Dimension 3', fontsize=12)
        ax.set_title(f'{title}\n({len(data)} points, {n_clusters} clusters)', fontsize=14, fontweight='bold')
        
        if show_labels:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"   📁 Plot saved to {save_path}")
    
    # Don't show plot interactively to avoid blocking
    plt.close()

def plot_cluster_comparison(analyzer, 
                           figsize: Tuple[int, int] = (16, 6),
                           save_path: Optional[str] = None) -> None:
    """
    Create side-by-side comparison of 2D and 3D UMAP clustering.
    
    Args:
        analyzer: DataExtractorAnalyzer instance with data loaded
        figsize: Figure size tuple
        save_path: Path to save the plot
    """
    if analyzer.embeddings is None:
        raise ValueError("No embeddings found. Please extract data first.")
    
    fig = plt.figure(figsize=figsize)
    
    # 2D Plot
    ax1 = fig.add_subplot(121)
    
    # Reduce to 2D and cluster
    from sklearn.preprocessing import StandardScaler
    import umap
    import hdbscan
    
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(analyzer.embeddings)
    
    reducer_2d = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    reduced_2d = reducer_2d.fit_transform(embeddings_scaled)
    
    clusterer_2d = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=3)
    labels_2d = clusterer_2d.fit_predict(reduced_2d)
    
    # Plot 2D
    unique_labels = sorted(set(labels_2d))
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels_2d == label
        if label == -1:
            ax1.scatter(reduced_2d[mask, 0], reduced_2d[mask, 1], 
                       c='lightgray', alpha=0.5, s=30, label='Noise')
        else:
            ax1.scatter(reduced_2d[mask, 0], reduced_2d[mask, 1], 
                       c=[colors[i]], alpha=0.7, s=60, label=f'Cluster {label}')
    
    ax1.set_xlabel('UMAP Dimension 1')
    ax1.set_ylabel('UMAP Dimension 2')
    ax1.set_title('2D UMAP Clustering')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 3D Plot
    ax2 = fig.add_subplot(122, projection='3d')
    
    reducer_3d = umap.UMAP(n_components=3, random_state=42, n_neighbors=15, min_dist=0.1)
    reduced_3d = reducer_3d.fit_transform(embeddings_scaled)
    
    clusterer_3d = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=3)
    labels_3d = clusterer_3d.fit_predict(reduced_3d)
    
    # Plot 3D
    unique_labels_3d = sorted(set(labels_3d))
    colors_3d = plt.cm.Set3(np.linspace(0, 1, len(unique_labels_3d)))
    
    for i, label in enumerate(unique_labels_3d):
        mask = labels_3d == label
        if label == -1:
            ax2.scatter(reduced_3d[mask, 0], reduced_3d[mask, 1], reduced_3d[mask, 2],
                       c='lightgray', alpha=0.5, s=30, label='Noise')
        else:
            ax2.scatter(reduced_3d[mask, 0], reduced_3d[mask, 1], reduced_3d[mask, 2],
                       c=[colors_3d[i]], alpha=0.7, s=60, label=f'Cluster {label}')
    
    ax2.set_xlabel('UMAP Dimension 1')
    ax2.set_ylabel('UMAP Dimension 2')
    ax2.set_zlabel('UMAP Dimension 3')
    ax2.set_title('3D UMAP Clustering')
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"   📁 Comparison plot saved to {save_path}")
    
    plt.close()

def plot_cluster_summary_chart(cluster_summary: pd.DataFrame,
                              figsize: Tuple[int, int] = (10, 6),
                              save_path: Optional[str] = None) -> None:
    """
    Create a bar chart showing cluster sizes.
    
    Args:
        cluster_summary: DataFrame with cluster summary information
        figsize: Figure size tuple
        save_path: Path to save the plot
    """
    # Filter out noise for the main chart
    main_clusters = cluster_summary[cluster_summary['cluster_id'] >= 0].copy()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Bar chart of cluster sizes
    bars = ax1.bar(range(len(main_clusters)), main_clusters['size'], 
                   color=plt.cm.Set3(np.linspace(0, 1, len(main_clusters))))
    
    ax1.set_xlabel('Cluster ID')
    ax1.set_ylabel('Number of Points')
    ax1.set_title('Cluster Sizes')
    ax1.set_xticks(range(len(main_clusters)))
    ax1.set_xticklabels([f'C{int(cid)}' for cid in main_clusters['cluster_id']])
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}', ha='center', va='bottom')
    
    # Pie chart of cluster distribution
    sizes = main_clusters['size'].tolist()
    labels = [f'Cluster {int(cid)}' for cid in main_clusters['cluster_id']]
    
    # Add noise if present
    noise_data = cluster_summary[cluster_summary['cluster_id'] == -1]
    if not noise_data.empty:
        sizes.append(noise_data['size'].iloc[0])
        labels.append('Noise')
    
    ax2.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90,
            colors=plt.cm.Set3(np.linspace(0, 1, len(sizes))))
    ax2.set_title('Cluster Distribution')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"   📁 Summary chart saved to {save_path}")
    
    plt.close()

def qualitative_verbal_evaluation(qualitative):
    if qualitative >= 0.8:
        return "Clusters are highly coherent, business-relevant, and easily interpretable. Excellent qualitative quality."
    elif qualitative >= 0.6:
        return "Clusters are generally coherent and business-relevant, with good interpretability. Qualitative quality is strong."
    elif qualitative >= 0.4:
        return "Clusters show moderate coherence and relevance. Some clusters may need review or merging for better interpretability."
    else:
        return "Clusters have low semantic coherence or business relevance. Consider reviewing clustering parameters or input data."

def write_clustering_report_md(collection_name, quantitative, qualitative, combined, cluster_count, noise_pct, silhouette, output_dir="."):
    """
    Write a Markdown report summarizing clustering results, including a verbal evaluation.
    """
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = f"{output_dir}/clustering_report_{collection_name}_{now.replace(' ', '_').replace(':', '-')}.md"
    with open(filename, "w") as f:
        f.write(f"# Clustering Report for `{collection_name}`\n")
        f.write(f"_Generated: {now}_\n\n")
        f.write("## Quantitative Measures\n")
        f.write(f"- **Combined Quantitative Score:** `{quantitative:.2f}`\n")
        f.write(f"- **Silhouette Score:** `{silhouette:.2f}`\n")
        f.write(f"- **Number of Clusters:** `{cluster_count}`\n")
        f.write(f"- **Noise Percentage:** `{noise_pct:.1f}%`\n\n")
        f.write("## Qualitative Measures\n")
        f.write(f"- **Combined Qualitative Score:** `{qualitative:.2f}`\n")
        f.write("\n## Overall Combined Score\n")
        f.write(f"- **Combined Score:** `{combined:.2f}`\n\n")
        f.write("## Verbal Evaluation\n")
        f.write(qualitative_verbal_evaluation(qualitative) + "\n")
    print(f"✅ Markdown report written to: {filename}")

def create_all_plots(analyzer, 
                    save_2d: str = "umap_2d_clusters.png",
                    save_3d: str = "umap_3d_clusters.png", 
                    save_comparison: str = "umap_comparison.png",
                    save_summary: str = "cluster_summary.png") -> None:
    """
    Create all visualization plots for the analyzer data.
    
    Args:
        analyzer: DataExtractorAnalyzer instance with processed data
        save_2d: Path for 2D plot
        save_3d: Path for 3D plot
        save_comparison: Path for comparison plot
        save_summary: Path for summary chart
    """
    if analyzer.data is None or analyzer.reduced_embeddings is None or analyzer.cluster_labels is None:
        raise ValueError("Analyzer must have data, reduced embeddings, and cluster labels")
    
    print("🎨 Creating visualization plots...")
    
    # 2D plot
    if analyzer.reduced_embeddings.shape[1] >= 2:
        plot_umap_clusters(
            analyzer.data, 
            analyzer.reduced_embeddings[:, :2], 
            analyzer.cluster_labels,
            save_path=save_2d,
            title="2D UMAP Clustering Results"
        )
    
    # 3D plot if we have 3D data
    if analyzer.reduced_embeddings.shape[1] >= 3:
        plot_umap_clusters(
            analyzer.data, 
            analyzer.reduced_embeddings, 
            analyzer.cluster_labels,
            save_path=save_3d,
            title="3D UMAP Clustering Results"
        )
    
    # Comparison plot
    plot_cluster_comparison(analyzer, save_path=save_comparison)
    
    # Summary chart
    cluster_summary = analyzer.get_cluster_summary()
    plot_cluster_summary_chart(cluster_summary, save_path=save_summary)
    
    print("   ✅ All plots created successfully!")
