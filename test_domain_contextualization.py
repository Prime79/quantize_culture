#!/usr/bin/env python3
"""
Test domain contextualization: "Domain Logic example phrase:" prefix vs original sentences.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
import numpy as np
from collections import defaultdict
from sklearn.metrics import silhouette_score
import umap
import hdbscan
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

def test_domain_contextualization():
    """Test if 'Domain Logic example phrase:' prefix improves clustering quality."""
    print("🧪 TESTING DOMAIN CONTEXTUALIZATION")
    print("=" * 60)
    print("Comparing: Original vs 'Domain Logic example phrase:' prefix")
    print()
    
    # Load sample sentences from data_01.json
    print("📊 Loading sentences from data_01.json...")
    with open('data_01.json', 'r') as f:
        data = json.load(f)
    
    # Extract first 30 sentences for speed
    sentences = []
    def extract_sentences(obj, count_limit=30):
        if len(sentences) >= count_limit:
            return
        if isinstance(obj, dict):
            for key, value in obj.items():
                extract_sentences(value, count_limit)
        elif isinstance(obj, list):
            for item in obj:
                if len(sentences) >= count_limit:
                    break
                sentences.append(item)
    
    extract_sentences(data)
    sentences = sentences[:30]  # Limit for speed
    print(f"✅ Loaded {len(sentences)} sentences for testing")
    
    # Create contextualized versions
    original_sentences = sentences
    contextualized_sentences = [f"Domain Logic example phrase: {sentence}" for sentence in sentences]
    
    print(f"\nExample transformation:")
    print(f"Original: {original_sentences[0]}")
    print(f"Contextualized: {contextualized_sentences[0]}")
    
    # Test both versions
    results = {}
    
    for version_name, test_sentences in [
        ("Original", original_sentences),
        ("Domain_Contextualized", contextualized_sentences)
    ]:
        print(f"\n{'='*20} TESTING {version_name.upper()} {'='*20}")
        
        # Create embeddings using our embedding function
        print(f"🔄 Creating embeddings for {len(test_sentences)} sentences...")
        
        # Use the same embedding approach as our main system
        import openai
        from dotenv import load_dotenv
        
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
        def get_embedding(text):
            """Get embedding for a single text."""
            response = openai.embeddings.create(
                input=text,
                model="text-embedding-3-small"
            )
            return response.data[0].embedding
        
        embeddings = []
        
        for i, sentence in enumerate(test_sentences):
            if i % 10 == 0:
                print(f"   Processing sentence {i+1}/{len(test_sentences)}")
            try:
                embedding = get_embedding(sentence)
                embeddings.append(embedding)
            except Exception as e:
                print(f"   ❌ Failed to embed sentence {i}: {e}")
                continue
        
        embeddings = np.array(embeddings)
        print(f"✅ Created {len(embeddings)} embeddings (shape: {embeddings.shape})")
        
        if len(embeddings) < 10:
            print("❌ Not enough embeddings for clustering")
            continue
        
        # Apply clustering with optimized parameters
        print("🔧 Running clustering...")
        
        # UMAP reduction
        reducer = umap.UMAP(
            n_neighbors=min(8, len(embeddings)-1),
            min_dist=0.01,
            n_components=3,
            metric='cosine',
            random_state=42
        )
        reduced_embeddings = reducer.fit_transform(embeddings)
        
        # HDBSCAN clustering
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=max(2, len(embeddings)//15),
            min_samples=1,
            metric='euclidean'
        )
        cluster_labels = clusterer.fit_predict(reduced_embeddings)
        
        # Calculate metrics
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        noise_count = (cluster_labels == -1).sum()
        noise_pct = noise_count / len(cluster_labels) * 100
        
        print(f"   Clusters Found: {n_clusters}")
        print(f"   Noise Points: {noise_count} ({noise_pct:.1f}%)")
        
        # Calculate quantitative scores
        quant_score = 0
        sil_score = 0
        
        if n_clusters > 1:
            valid_mask = cluster_labels != -1
            if valid_mask.sum() > 1:
                sil_score = silhouette_score(reduced_embeddings[valid_mask], cluster_labels[valid_mask])
                quant_score = (sil_score + 1) * 5  # Convert to 0-10 scale
                if noise_pct > 20:
                    quant_score *= 0.8
                
                print(f"   Silhouette Score: {sil_score:.3f}")
                print(f"   📊 Quantitative Score: {quant_score:.1f}/10")
        
        # Calculate qualitative scores (simplified for speed)
        qual_score = 0
        
        if n_clusters > 0:
            print("   🎨 Calculating qualitative scores...")
            try:
                from app.qualitative_assessment import QualitativeClusteringAssessment
                
                # Group sentences by cluster
                clusters = defaultdict(list)
                for i, (sentence, label) in enumerate(zip(test_sentences, cluster_labels)):
                    if label != -1:  # Skip noise
                        clusters[label].append(sentence)
                
                # Assess clusters (limit to 5 largest for speed)
                assessor = QualitativeClusteringAssessment()
                cluster_sizes = [(k, len(v)) for k, v in clusters.items()]
                top_clusters = sorted(cluster_sizes, key=lambda x: x[1], reverse=True)[:5]
                
                total_qual = 0
                assessed_count = 0
                
                for cluster_id, size in top_clusters:
                    if size >= 2:  # Need multiple sentences
                        cluster_sentences = clusters[cluster_id][:3]  # Limit for speed
                        try:
                            assessment = assessor.assess_single_cluster_quality(cluster_sentences, str(cluster_id))
                            total_qual += assessment['overall_qualitative_score']
                            assessed_count += 1
                        except Exception as e:
                            print(f"      Cluster {cluster_id} assessment failed: {e}")
                
                if assessed_count > 0:
                    qual_score = total_qual / assessed_count
                    print(f"   📊 Qualitative Score: {qual_score:.2f}/1.0")
                else:
                    print("   ❌ No clusters assessed for qualitative score")
                    
            except Exception as e:
                print(f"   ❌ Qualitative assessment failed: {e}")
        
        # Combined score
        qual_score_scaled = qual_score * 10
        combined_score = (quant_score * 0.4) + (qual_score_scaled * 0.6)
        
        print(f"   🎯 Combined Score: {combined_score:.1f}/10")
        
        # Store results
        results[version_name] = {
            'n_clusters': n_clusters,
            'noise_pct': noise_pct,
            'silhouette_score': sil_score,
            'quantitative_score': quant_score,
            'qualitative_score': qual_score,
            'combined_score': combined_score,
            'cluster_labels': cluster_labels,
            'sentences': test_sentences
        }
    
    # Compare results
    print(f"\n🏆 COMPARISON RESULTS")
    print("=" * 50)
    
    if len(results) == 2:
        orig = results['Original']
        context = results['Domain_Contextualized']
        
        print(f"📊 METRICS COMPARISON:")
        print(f"   Clusters Found:")
        print(f"     Original: {orig['n_clusters']}")
        print(f"     Contextualized: {context['n_clusters']}")
        
        print(f"   Noise Percentage:")
        print(f"     Original: {orig['noise_pct']:.1f}%")
        print(f"     Contextualized: {context['noise_pct']:.1f}%")
        
        print(f"   Quantitative Score:")
        print(f"     Original: {orig['quantitative_score']:.1f}/10")
        print(f"     Contextualized: {context['quantitative_score']:.1f}/10")
        print(f"     Change: {context['quantitative_score'] - orig['quantitative_score']:+.1f}")
        
        print(f"   Qualitative Score:")
        print(f"     Original: {orig['qualitative_score']:.2f}/1.0")
        print(f"     Contextualized: {context['qualitative_score']:.2f}/1.0") 
        print(f"     Change: {context['qualitative_score'] - orig['qualitative_score']:+.2f}")
        
        print(f"   Combined Score:")
        print(f"     Original: {orig['combined_score']:.1f}/10")
        print(f"     Contextualized: {context['combined_score']:.1f}/10")
        print(f"     Change: {context['combined_score'] - orig['combined_score']:+.1f}")
        
        # Conclusion
        improvement = context['combined_score'] - orig['combined_score']
        
        print(f"\n🎯 CONCLUSION:")
        if improvement > 0.5:
            print(f"   🎉 SIGNIFICANT IMPROVEMENT: +{improvement:.1f} points!")
            print(f"   ✅ 'Domain Logic example phrase:' prefix improves clustering quality")
        elif improvement > 0:
            print(f"   ✅ MODEST IMPROVEMENT: +{improvement:.1f} points")
            print(f"   👍 'Domain Logic example phrase:' prefix helps slightly")
        elif improvement < -0.5:
            print(f"   ❌ SIGNIFICANT DECREASE: {improvement:.1f} points")
            print(f"   👎 'Domain Logic example phrase:' prefix hurts clustering quality")
        else:
            print(f"   ➡️  NO SIGNIFICANT CHANGE: {improvement:.1f} points")
            print(f"   🤷 'Domain Logic example phrase:' prefix has minimal effect")
        
        return orig, context
    else:
        print("❌ Could not complete comparison - insufficient results")
        return None, None

if __name__ == "__main__":
    test_domain_contextualization()
