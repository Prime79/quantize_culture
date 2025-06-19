#!/usr/bin/env python3
"""
Simplified domain contextualization test using the working clustering pattern
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

def test_domain_context_simplified():
    """
    Simplified test of domain contextualization using working patterns
    """
    print("🧪 DOMAIN CONTEXTUALIZATION TEST - SIMPLIFIED")
    print("=" * 55)
    print("Testing: 'Domain Logic example phrase:' prefix")
    print()
    
    # Create contextualized data file
    print("1️⃣  Creating contextualized data...")
    with open('data_01.json', 'r') as f:
        original_data = json.load(f)
    
    contextualized_data = {}
    for category, subcategories in original_data.items():
        contextualized_data[category] = {}
        for subcategory, sentences in subcategories.items():
            contextualized_data[category][subcategory] = [
                f"Domain Logic example phrase: {sentence}" 
                for sentence in sentences
            ]
    
    with open('data_01_contextualized.json', 'w') as f:
        json.dump(contextualized_data, f, indent=2)
    
    print("✅ Created contextualized version")
    
    # Load and embed both versions
    print("\n2️⃣  Loading and embedding both versions...")
    
    # Load original data
    from app.load_data import load_sentences_from_json
    original_sentences = load_sentences_from_json('data_01.json')
    contextualized_sentences = load_sentences_from_json('data_01_contextualized.json')
    
    print(f"Original: {len(original_sentences)} sentences")
    print(f"Contextualized: {len(contextualized_sentences)} sentences")
    
    # Store in separate collections
    from app.embed_and_store import embed_and_store_bulk
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import VectorParams, Distance
    
    client = QdrantClient(host="localhost", port=6333)
    
    # Create collections
    collections_to_create = [
        ("context_test_original", original_sentences),
        ("context_test_contextualized", contextualized_sentences)
    ]
    
    for collection_name, sentences in collections_to_create:
        # Recreate collection
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )
        
        # Embed and store
        embed_and_store_bulk(sentences, qdrant_client=client, collection_name=collection_name)
        print(f"✅ Loaded {len(sentences)} sentences to {collection_name}")
    
    # Extract embeddings using our fixed extractor
    print("\n3️⃣  Extracting embeddings...")
    
    from app.extract import DataExtractorAnalyzer
    
    # Original embeddings
    extractor_orig = DataExtractorAnalyzer(collection_name="context_test_original")
    df_orig = extractor_orig.extract_data(limit=None)
    embeddings_orig = extractor_orig.embeddings
    sentences_orig = df_orig['sentence'].tolist()
    
    # Contextualized embeddings  
    extractor_ctx = DataExtractorAnalyzer(collection_name="context_test_contextualized")
    df_ctx = extractor_ctx.extract_data(limit=None)
    embeddings_ctx = extractor_ctx.embeddings
    sentences_ctx = df_ctx['sentence'].tolist()
    
    print(f"✅ Original embeddings: {embeddings_orig.shape}")
    print(f"✅ Contextualized embeddings: {embeddings_ctx.shape}")
    
    # Run clustering on both
    print("\n4️⃣  Running clustering on both versions...")
    
    def cluster_and_score(embeddings, sentences, name):
        """Run clustering and calculate scores"""
        print(f"\\n--- {name} ---")
        
        # UMAP reduction
        reducer = umap.UMAP(
            n_neighbors=min(15, len(embeddings)-1),
            min_dist=0.01,
            n_components=3,
            metric='cosine',
            random_state=42
        )
        reduced = reducer.fit_transform(embeddings)
        
        # HDBSCAN clustering
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=max(3, len(embeddings)//40),
            min_samples=2,
            metric='euclidean'
        )
        labels = clusterer.fit_predict(reduced)
        
        # Calculate quantitative scores
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_count = (labels == -1).sum()
        noise_pct = noise_count / len(labels) * 100
        
        quant_score = 0
        if n_clusters > 1:
            valid_mask = labels != -1
            if valid_mask.sum() > 1:
                sil_score = silhouette_score(reduced[valid_mask], labels[valid_mask])
                quant_score = (sil_score + 1) * 5  # Convert to 0-10 scale
                if noise_pct > 20:
                    quant_score *= 0.8
        
        print(f"   Clusters: {n_clusters}")
        print(f"   Noise: {noise_pct:.1f}%")
        print(f"   Quantitative Score: {quant_score:.1f}/10")
        
        # Quick qualitative assessment on largest clusters
        qual_score = 0
        try:
            from app.qualitative_assessment import QualitativeClusteringAssessment
            
            # Group sentences by cluster
            clusters = defaultdict(list)
            for sentence, label in zip(sentences, labels):
                if label != -1:  # Skip noise
                    clusters[label].append(sentence)
            
            if len(clusters) > 0:
                assessor = QualitativeClusteringAssessment()
                
                # Assess top 3 largest clusters
                cluster_sizes = [(k, len(v)) for k, v in clusters.items()]
                top_clusters = sorted(cluster_sizes, key=lambda x: x[1], reverse=True)[:3]
                
                scores = []
                for cluster_id, size in top_clusters:
                    if size >= 2:
                        cluster_sentences = clusters[cluster_id][:5]  # Limit for speed
                        assessment = assessor.assess_single_cluster_quality(cluster_sentences, str(cluster_id))
                        scores.append(assessment['overall_qualitative_score'])
                
                if scores:
                    qual_score = np.mean(scores) * 10  # Scale to 0-10
                    print(f"   Qualitative Score: {qual_score:.1f}/10")
                else:
                    print(f"   Qualitative Score: N/A (no clusters to assess)")
            else:
                print(f"   Qualitative Score: N/A (no clusters)")
                
        except Exception as e:
            print(f"   Qualitative Score: Error - {str(e)[:50]}...")
        
        # Combined score (40% quant, 60% qual)
        if qual_score > 0 and quant_score > 0:
            combined = (quant_score * 0.4) + (qual_score * 0.6)
            print(f"   Combined Score: {combined:.1f}/10")
        else:
            combined = quant_score if quant_score > 0 else 0
            print(f"   Combined Score: {combined:.1f}/10 (quantitative only)")
        
        return {
            'n_clusters': n_clusters,
            'noise_pct': noise_pct,
            'quantitative': quant_score,
            'qualitative': qual_score,
            'combined': combined
        }
    
    # Test both versions
    result_original = cluster_and_score(embeddings_orig, sentences_orig, "ORIGINAL SENTENCES")
    result_contextualized = cluster_and_score(embeddings_ctx, sentences_ctx, "CONTEXTUALIZED SENTENCES")
    
    # Compare results
    print(f"\n📊 COMPARISON RESULTS")
    print("=" * 50)
    
    improvements = {
        'quantitative': result_contextualized['quantitative'] - result_original['quantitative'],
        'qualitative': result_contextualized['qualitative'] - result_original['qualitative'],
        'combined': result_contextualized['combined'] - result_original['combined']
    }
    
    print(f"IMPROVEMENTS with 'Domain Logic example phrase:' prefix:")
    for metric, improvement in improvements.items():
        emoji = "🎉" if improvement > 0.5 else "✅" if improvement > 0 else "❌" if improvement < -0.5 else "➡️"
        print(f"   {metric.capitalize()}: {improvement:+.1f} {emoji}")
    
    # Conclusion
    if improvements['combined'] > 0.5:
        print(f"\n🎉 SIGNIFICANT IMPROVEMENT!")
        print(f"   Domain contextualization helps clustering quality!")
    elif improvements['combined'] > 0:
        print(f"\n✅ Slight improvement with domain contextualization")
    elif improvements['combined'] < -0.5:
        print(f"\n❌ Domain contextualization hurt performance")
    else:
        print(f"\n➡️  No significant change")
    
    return result_original, result_contextualized

if __name__ == "__main__":
    test_domain_context_simplified()
