#!/usr/bin/env python3
"""
Complete demo showing both QUANTITATIVE and QUALITATIVE clustering scores.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from collections import defaultdict
from sklearn.metrics import silhouette_score
import umap
import hdbscan

def run_complete_scoring_demo():
    """Run complete demo with both quantitative and qualitative scores."""
    print("🚀 Complete Clustering Scores Demo")
    print("=" * 50)
    print("Showing both QUANTITATIVE and QUALITATIVE measures")
    print()
    
    # Extract limited data for speed
    from app.extract import DataExtractorAnalyzer
    
    print("📊 Loading 50 records for quick demo...")
    analyzer = DataExtractorAnalyzer()
    df = analyzer.extract_data(limit=50)
    embeddings = analyzer.embeddings
    sentences = df['sentence'].tolist()
    
    print(f"✅ Got {len(embeddings)} embeddings and sentences")
    
    # Quick clustering with optimized parameters
    print("\n🔧 Running clustering...")
    print("   Using Ultra_Fine parameters (best performing)")
    
    # UMAP reduction
    reducer = umap.UMAP(
        n_neighbors=6,
        min_dist=0.001,
        n_components=3,
        metric='cosine',
        random_state=42
    )
    reduced_embeddings = reducer.fit_transform(embeddings)
    
    # HDBSCAN clustering  
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=2,
        min_samples=1,
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
            if n_clusters < 3:
                quant_score *= 0.9  # Slight penalty for too few clusters
                
            print(f"   📊 Quantitative Score: {quant_score:.1f}/10")
        else:
            quant_score = 0
            print("   ❌ Not enough valid points for silhouette score")
    else:
        quant_score = 0
        print("   ❌ Not enough clusters for meaningful metrics")
    
    # QUALITATIVE SCORES (Semantic/Cultural)
    print(f"\n🎨 QUALITATIVE SCORES (Semantic/Cultural):")
    print("-" * 50)
    
    if n_clusters > 0:
        from app.qualitative_assessment import QualitativeClusteringAssessment
        
        try:
            assessor = QualitativeClusteringAssessment()
            total_qual_score = 0
            assessed_clusters = 0
            cluster_themes = []
            
            # Assess clusters (limit to first 5 for speed)
            cluster_ids = [k for k in clusters.keys() if k != -1][:5]
            
            for cluster_id in cluster_ids:
                cluster_sentences = clusters[cluster_id]
                if len(cluster_sentences) >= 2:  # Need multiple sentences
                    try:
                        assessment = assessor.assess_single_cluster_quality(
                            cluster_sentences[:3], str(cluster_id)  # Limit sentences for speed
                        )
                        qual_score = assessment['overall_qualitative_score']
                        theme = assessment['interpretability']['theme_name']
                        business_value = assessment['interpretability']['business_value']
                        
                        print(f"   Cluster {cluster_id}: {qual_score:.3f} - '{theme}' (BV: {business_value:.2f})")
                        
                        total_qual_score += qual_score
                        assessed_clusters += 1
                        cluster_themes.append(theme)
                        
                    except Exception as e:
                        print(f"   Cluster {cluster_id}: Assessment error - {str(e)[:40]}...")
            
            if assessed_clusters > 0:
                avg_qual_score = total_qual_score / assessed_clusters
                print(f"   📊 Qualitative Score: {avg_qual_score:.1f}/1.0 (avg)")
                
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
                
                print(f"\n📋 TOP THEMES IDENTIFIED:")
                for i, theme in enumerate(cluster_themes[:3], 1):
                    print(f"   {i}. {theme}")
                
                # Recommendations
                print(f"\n💡 QUICK RECOMMENDATIONS:")
                if quant_score < 5:
                    print("   • Consider adjusting clustering parameters (quantitative)")
                if qual_score_scaled < 5:
                    print("   • Review data quality and cultural relevance (qualitative)")
                if noise_pct > 25:
                    print("   • High noise level - consider relaxing min_cluster_size")
                if n_clusters > 20:
                    print("   • Many small clusters - consider increasing min_cluster_size")
                    
            else:
                print("   ❌ No clusters large enough for qualitative assessment")
                
        except Exception as e:
            print(f"   ❌ Qualitative assessment error: {e}")
            print("   (This may be due to missing OpenAI API key)")
    else:
        print("   ❌ No clusters found for qualitative assessment")
    
    print(f"\n✅ Demo completed!")
    print(f"📊 Results: {n_clusters} clusters, {quant_score:.1f}/10 quantitative score")

if __name__ == "__main__":
    run_complete_scoring_demo()
