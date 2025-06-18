#!/usr/bin/env python3
"""
Quick scoring demo with limited data to avoid long database queries.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def quick_scoring_demo():
    """Run a quick demo with limited data to show both quantitative and qualitative scores."""
    print("🚀 Quick Scoring Demo - Limited Data")
    print("=" * 50)
    
    # Import with limit
    from app.extract import DataExtractorAnalyzer
    
    print("📊 Extracting limited data (50 records)...")
    analyzer = DataExtractorAnalyzer()
    
    # Extract only 50 records for speed
    df = analyzer.extract_data(limit=50)
    
    if df is None or df.empty:
        print("❌ No data found")
        return
    
    print(f"✅ Extracted {len(df)} sentences")
    
    # Check if we have existing clusters
    if 'cluster' in df.columns and df['cluster'].notna().any():
        print("🎯 Found existing clusters - running assessment...")
        
        # Quick qualitative assessment on existing clusters
        from collections import defaultdict
        clusters = defaultdict(list)
        for _, row in df.iterrows():
            cluster_id = row.get('cluster', -1)
            sentence = row.get('sentence', '')
            if sentence and cluster_id != -1:
                clusters[cluster_id].append(sentence)
        
        print(f"📈 Found {len(clusters)} clusters to assess")
        
        # Quick qualitative assessment (limited)
        if len(clusters) > 0:
            from app.qualitative_assessment import QualitativeClusteringAssessment
            assessor = QualitativeClusteringAssessment()
            
            print("\n🎨 QUALITATIVE SCORES (Semantic/Cultural):")
            print("-" * 45)
            
            total_qual_score = 0
            cluster_count = 0
            
            # Assess first 3 clusters for speed
            for cluster_id, sentences in list(clusters.items())[:3]:
                if len(sentences) >= 2:  # Only assess clusters with multiple sentences
                    try:
                        assessment = assessor.assess_single_cluster_quality(sentences[:5], cluster_id)  # Limit sentences
                        qual_score = assessment['overall_qualitative_score']
                        theme = assessment['interpretability']['theme_name']
                        
                        print(f"  Cluster {cluster_id}: {qual_score:.3f} - {theme}")
                        total_qual_score += qual_score
                        cluster_count += 1
                    except Exception as e:
                        print(f"  Cluster {cluster_id}: Error - {str(e)[:50]}...")
            
            if cluster_count > 0:
                avg_qual_score = total_qual_score / cluster_count
                print(f"\n📊 Average Qualitative Score: {avg_qual_score:.3f}")
            
        # Show quantitative scores if available
        print(f"\n📈 QUANTITATIVE SCORES (Mathematical/Statistical):")
        print("-" * 50)
        if 'cluster' in df.columns:
            from sklearn.metrics import silhouette_score
            import numpy as np
            
            # Get embeddings and cluster labels
            embeddings = np.array([row['embedding'] for _, row in df.iterrows() if row['cluster'] != -1])
            labels = [row['cluster'] for _, row in df.iterrows() if row['cluster'] != -1]
            
            if len(set(labels)) > 1 and len(embeddings) > 1:
                try:
                    sil_score = silhouette_score(embeddings, labels)
                    noise_pct = (df['cluster'] == -1).sum() / len(df) * 100
                    n_clusters = len(set(labels))
                    
                    print(f"  Silhouette Score: {sil_score:.3f}")
                    print(f"  Clusters Found: {n_clusters}")  
                    print(f"  Noise Percentage: {noise_pct:.1f}%")
                    
                    # Simple quantitative score (0-10 scale)
                    quant_score = (sil_score + 1) * 5  # Convert -1,1 to 0,10
                    if noise_pct > 20:
                        quant_score *= 0.8  # Penalty for high noise
                    
                    print(f"  Quantitative Score: {quant_score:.3f}")
                    
                    if cluster_count > 0:
                        # Combined score (40% quant, 60% qual)
                        combined = (quant_score * 0.4) + (avg_qual_score * 10 * 0.6)
                        print(f"\n🏆 COMBINED SCORE: {combined:.3f}")
                        print(f"    (40% Quantitative + 60% Qualitative)")
                        
                except Exception as e:
                    print(f"  Error calculating quantitative scores: {e}")
            else:
                print("  Not enough clusters for quantitative assessment")
        else:
            print("  No cluster data available for quantitative assessment")
            
    else:
        print("🔧 No existing clusters found")
        print("   Run clustering first with: python improve_clustering.py")
        
        # Show data info
        print(f"\n📋 Data Summary:")
        print(f"   Total sentences: {len(df)}")
        if 'embedding' in df.columns:
            print(f"   Embeddings available: ✅")
        else:
            print(f"   Embeddings available: ❌")

if __name__ == "__main__":
    quick_scoring_demo()
