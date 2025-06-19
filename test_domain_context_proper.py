#!/usr/bin/env python3
"""
Test domain contextualization using EXISTING workflow - no recreation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.load_data import load_sentences_from_json
from app.clustering_optimizer import EnhancedDataExtractorAnalyzer
from qdrant_client import QdrantClient
import json

def test_domain_context_using_existing_workflow():
    """
    Test domain contextualization using EXISTING workflow - no recreation
    """
    print("🧪 TESTING DOMAIN CONTEXTUALIZATION")
    print("Using existing codebase workflow")
    print("=" * 50)
    
    # Load original data using existing function
    sentences = load_sentences_from_json('data_01.json')
    print(f"Loaded {len(sentences)} sentences from data_01.json")
    
    # Create modified JSON with domain context prefixes
    print("\n1️⃣  Creating contextualized version of data_01.json...")
    
    # Load original structured data
    with open('data_01.json', 'r') as f:
        original_data = json.load(f)
    
    # Create contextualized version
    contextualized_data = {}
    for category, subcategories in original_data.items():
        contextualized_data[category] = {}
        for subcategory, sentences in subcategories.items():
            contextualized_data[category][subcategory] = [
                f"Domain Logic example phrase: {sentence}" 
                for sentence in sentences
            ]
    
    # Save contextualized version
    with open('data_01_contextualized.json', 'w') as f:
        json.dump(contextualized_data, f, indent=2)
    
    print("✅ Created data_01_contextualized.json")
    
    # Use existing workflow but create our own loading function that accepts collection name
    print("\n2️⃣  Loading original data using existing workflow...")
    
    def load_data_with_collection_name(json_file, collection_name):
        """Load data to specific collection using existing functions"""
        from app.embed_and_store import embed_and_store_bulk
        from qdrant_client import QdrantClient
        from qdrant_client.http.models import VectorParams, Distance
        
        # Load sentences using existing function
        sentences = load_sentences_from_json(json_file)
        
        # Create client and collection
        client = QdrantClient(host="localhost", port=6333)
        
        # Create collection if it doesn't exist
        collections = client.get_collections().collections
        if not any(c.name == collection_name for c in collections):
            client.recreate_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
            )
            print(f"Created collection '{collection_name}'")
        
        # Use existing embed_and_store_bulk function
        embed_and_store_bulk(sentences, qdrant_client=client, collection_name=collection_name)
        
        return len(sentences)
    
    # Load original data
    try:
        count_original = load_data_with_collection_name('data_01.json', 'culture_original_test')
        print(f"✅ Loaded {count_original} original sentences")
    except Exception as e:
        print(f"Original data loading error: {e}")
    
    print("\n3️⃣  Loading contextualized data using existing workflow...")
    
    # Load contextualized version
    count_contextualized = load_data_with_collection_name('data_01_contextualized.json', 'culture_contextualized_test')
    print(f"✅ Loaded {count_contextualized} contextualized sentences")
    
    # Use existing assessment workflow
    analyzer = EnhancedDataExtractorAnalyzer()
    
    print("\n4️⃣  Running assessment on ORIGINAL data...")
    
    # Extract embeddings from original collection
    from app.extract import DataExtractorAnalyzer
    extractor_original = DataExtractorAnalyzer(collection_name='culture_original_test')
    df_original = extractor_original.extract_data(limit=None)
    embeddings_original = extractor_original.embeddings
    
    print(f"   Extracted {len(embeddings_original)} original embeddings")
    
    result_original = analyzer.run_comprehensive_assessment(
        embeddings=embeddings_original,
        include_qualitative=True
    )
    
    print("\n5️⃣  Running assessment on CONTEXTUALIZED data...")
    
    # Extract embeddings from contextualized collection
    extractor_contextualized = DataExtractorAnalyzer(collection_name='culture_contextualized_test')
    df_contextualized = extractor_contextualized.extract_data(limit=None)
    embeddings_contextualized = extractor_contextualized.embeddings
    
    print(f"   Extracted {len(embeddings_contextualized)} contextualized embeddings")
    
    result_contextualized = analyzer.run_comprehensive_assessment(
        embeddings=embeddings_contextualized,
        include_qualitative=True
    )
    
    # Compare results using existing result structure
    print("\n📊 COMPARISON RESULTS")
    print("-" * 50)
    print("ORIGINAL SENTENCES:")
    print(f"  Quantitative: {result_original['quantitative_assessment']['combined_score']:.3f}")
    print(f"  Qualitative: {result_original['qualitative_assessment']['combined_score']:.3f}")
    print(f"  Combined: {result_original['combined_assessment']['combined_score']:.3f}")
    
    print("\nCONTEXTUALIZED SENTENCES ('Domain Logic example phrase:'):")
    print(f"  Quantitative: {result_contextualized['quantitative_assessment']['combined_score']:.3f}")
    print(f"  Qualitative: {result_contextualized['qualitative_assessment']['combined_score']:.3f}")
    print(f"  Combined: {result_contextualized['combined_assessment']['combined_score']:.3f}")
    
    # Calculate improvements
    quant_diff = result_contextualized['quantitative_assessment']['combined_score'] - result_original['quantitative_assessment']['combined_score']
    qual_diff = result_contextualized['qualitative_assessment']['combined_score'] - result_original['qualitative_assessment']['combined_score']
    combined_diff = result_contextualized['combined_assessment']['combined_score'] - result_original['combined_assessment']['combined_score']
    
    print(f"\nIMPROVEMENTS:")
    print(f"  Quantitative: {quant_diff:+.3f}")
    print(f"  Qualitative: {qual_diff:+.3f}")
    print(f"  Combined: {combined_diff:+.3f}")
    
    if combined_diff > 0.1:
        print("\n🎉 SIGNIFICANT IMPROVEMENT with domain contextualization!")
    elif combined_diff > 0:
        print("\n✅ Slight improvement with domain contextualization")
    elif combined_diff < -0.1:
        print("\n❌ Domain contextualization hurt performance")
    else:
        print("\n➡️  No significant change")
    
    return result_original, result_contextualized

if __name__ == "__main__":
    test_domain_context_using_existing_workflow()
