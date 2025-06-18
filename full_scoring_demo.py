#!/usr/bin/env python3
"""
Complete scoring demo on ALL data from data_01.json (200 sentences).
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.metrics import silhouette_score
import umap
import hdbscan

# Import our fixed extraction function
from fixed_extract import extract_data_fixed

def run_full_scoring_demo():
    """Run complete demo with both quantitative and qualitative scores on ALL data."""
    print("🚀 Complete Clustering Scores - ALL DATA (200 sentences)")
    print("=" * 65)
    print("Showing both QUANTITATIVE and QUALITATIVE measures")
    print()
    
    # Extract ALL data using fixed method
    print("📊 Loading ALL 200 sentences from data_01.json...")
    df, embeddings = extract_data_fixed(limit=None)  # Get all data
    
    if df is None or embeddings is None:
        print("❌ Failed to extract data")
        return
    
    sentences = df['sentence'].tolist()
    print(f"✅ Got {len(embeddings)} embeddings and sentences")
    
    # Clustering with optimized parameters on full dataset
    print("\n🔧 Running clustering on full dataset...")
    print("   Using Ultra_Fine parameters (optimized for good performance)")
    
    # UMAP reduction - adjusted for larger dataset
    reducer = umap.UMAP(
        n_neighbors=min(15, len(embeddings)-1),  # Adjust for dataset size
        min_dist=0.01,
        n_components=3,
        metric='cosine',
        random_state=42
    )
    reduced_embeddings = reducer.fit_transform(embeddings)
    
    # HDBSCAN clustering - adjusted for larger dataset
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=max(3, len(embeddings)//40),  # Adaptive min cluster size
        min_samples=2,
        metric='euclidean'
    )
    cluster_labels = clusterer.fit_predict(reduced_embeddings)
    
    # Group sentences by cluster
    clusters = defaultdict(list)
    for i, (sentence, label) in enumerate(zip(sentences, cluster_labels)):
        clusters[label].append(sentence)
    
    n_clusters = len([k for k in clusters.keys() if k != -1])
    noise_count = len(clusters.get(-1, []))
    
    print(f"   ✅ Found {n_clusters} clusters, {noise_count} noise points")
    
    # QUANTITATIVE SCORES (Mathematical/Statistical)
    print(f"\n📈 QUANTITATIVE SCORES (Mathematical/Statistical):")
    print("-" * 55)
    
    noise_pct = noise_count / len(cluster_labels) * 100
    print(f"   Total Sentences: {len(sentences)}")
    print(f"   Clusters Found: {n_clusters}")
    print(f"   Noise Points: {noise_count} ({noise_pct:.1f}%)")
    
    if n_clusters > 1:
        # Calculate silhouette score (only for non-noise points)
        valid_mask = cluster_labels != -1
        if valid_mask.sum() > 1:
            sil_score = silhouette_score(reduced_embeddings[valid_mask], cluster_labels[valid_mask])
            print(f"   Silhouette Score: {sil_score:.3f}")
            
            # Calculate quantitative score (0-10 scale)
            quant_score = (sil_score + 1) * 5  # Convert from [-1,1] to [0,10]
            if noise_pct > 20:
                quant_score *= 0.8  # Penalty for high noise
            if n_clusters < 5:
                quant_score *= 0.9  # Slight penalty for too few clusters
            elif n_clusters > 50:
                quant_score *= 0.9  # Slight penalty for too many clusters
                
            print(f"   📊 Quantitative Score: {quant_score:.1f}/10")
            
            # Show cluster distribution
            cluster_sizes = {}
            for label in set(cluster_labels):
                if label != -1:
                    cluster_sizes[label] = (cluster_labels == label).sum()
            
            sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)
            print(f"   Largest Clusters: {dict(sorted_clusters[:5])}")
        else:
            quant_score = 0
            print("   ❌ Not enough valid points for silhouette score")
    else:
        quant_score = 0
        print("   ❌ Not enough clusters for meaningful metrics")
    
    # QUALITATIVE SCORES (Semantic/Cultural) - Limited for speed
    print(f"\n🎨 QUALITATIVE SCORES (Semantic/Cultural):")
    print("-" * 50)
    print("   (Assessing top 10 largest clusters for speed)")
    
    if n_clusters > 0:
        try:
            from app.qualitative_assessment import QualitativeClusteringAssessment
            
            assessor = QualitativeClusteringAssessment()
            total_qual_score = 0
            assessed_clusters = 0
            cluster_themes = []
            
            # Get top clusters by size (limit to 10 for speed)
            cluster_sizes = [(k, len(v)) for k, v in clusters.items() if k != -1]
            top_clusters = sorted(cluster_sizes, key=lambda x: x[1], reverse=True)[:10]
            
            print(f"   Assessing {len(top_clusters)} largest clusters...")
            
            for cluster_id, size in top_clusters:
                cluster_sentences = clusters[cluster_id]
                if len(cluster_sentences) >= 2:  # Need multiple sentences
                    try:
                        # Limit sentences per cluster for speed
                        sample_sentences = cluster_sentences[:5]
                        assessment = assessor.assess_single_cluster_quality(
                            sample_sentences, str(cluster_id)
                        )
                        qual_score = assessment['overall_qualitative_score']
                        theme = assessment['interpretability']['theme_name']
                        business_value = assessment['interpretability']['business_value']
                        
                        print(f"   Cluster {cluster_id} ({size} items): {qual_score:.3f} - '{theme}' (BV: {business_value:.2f})")
                        
                        total_qual_score += qual_score
                        assessed_clusters += 1
                        cluster_themes.append((theme, size))
                        
                    except Exception as e:
                        print(f"   Cluster {cluster_id}: Assessment error - {str(e)[:40]}...")
            
            if assessed_clusters > 0:
                avg_qual_score = total_qual_score / assessed_clusters
                print(f"   📊 Qualitative Score: {avg_qual_score:.2f}/1.0 (avg of {assessed_clusters} clusters)")
                
                # COMBINED SCORE
                print(f"\n🏆 COMBINED ASSESSMENT:")
                print("-" * 30)
                
                # Scale qualitative to 0-10 to match quantitative
                qual_score_scaled = avg_qual_score * 10
                
                # Weighted combination: 40% quantitative, 60% qualitative
                combined_score = (quant_score * 0.4) + (qual_score_scaled * 0.6)
                
                print(f"   Quantitative (40%): {quant_score:.1f}/10")
                print(f"   Qualitative (60%):  {qual_score_scaled:.1f}/10")
                print(f"   🎯 Combined Score:   {combined_score:.1f}/10")
                
                print(f"\n📋 TOP THEMES IDENTIFIED (by cluster size):")
                sorted_themes = sorted(cluster_themes, key=lambda x: x[1], reverse=True)
                for i, (theme, size) in enumerate(sorted_themes[:5], 1):
                    print(f"   {i}. {theme} ({size} statements)")
                
                # Performance analysis
                print(f"\n📊 PERFORMANCE ANALYSIS:")
                if combined_score >= 7:
                    grade = "Excellent"
                elif combined_score >= 6:
                    grade = "Good"
                elif combined_score >= 5:
                    grade = "Fair"
                else:
                    grade = "Needs Improvement"
                
                print(f"   Overall Grade: {grade}")
                print(f"   Dataset Size: {len(sentences)} sentences")
                print(f"   Clustering Efficiency: {(len(sentences) - noise_count) / len(sentences) * 100:.1f}% clustered")
                
                # Recommendations
                print(f"\n💡 RECOMMENDATIONS:")
                if quant_score < 6:
                    print("   • Consider adjusting clustering parameters for better mathematical performance")
                if qual_score_scaled < 6:
                    print("   • Review data quality and semantic coherence of clusters")
                if noise_pct > 25:
                    print("   • High noise level - consider relaxing clustering parameters")
                if n_clusters > 30:
                    print("   • Many small clusters - consider increasing min_cluster_size")
                if n_clusters < 5:
                    print("   • Few clusters - data might benefit from more granular analysis")
                    
            else:
                print("   ❌ No clusters large enough for qualitative assessment")
                
        except Exception as e:
            print(f"   ❌ Qualitative assessment error: {e}")
            print("   (This may be due to missing OpenAI API key or network issues)")
    else:
        print("   ❌ No clusters found for qualitative assessment")
    
    print(f"\n✅ Full dataset analysis completed!")
    print(f"📊 Final Results: {n_clusters} clusters from {len(sentences)} sentences")

if __name__ == "__main__":
    run_full_scoring_demo()
