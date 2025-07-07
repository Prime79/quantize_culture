#!/usr/bin/env python3
"""
Test DL inference with the enhanced collection that has DL metadata.
"""
from app.core.inference_engine import EnhancedInferenceEngine
from app.services.qdrant_client import QdrantService
from app.services.openai_client import OpenAIService

def test_dl_enhanced_inference():
    """
    Test inference using the collection with DL metadata.
    """
    print("🚀 Testing DL-Enhanced Inference")
    print("=" * 50)
    
    # Initialize services
    qdrant_service = QdrantService()
    openai_service = OpenAIService()
    inference_engine = EnhancedInferenceEngine(openai_service, qdrant_service)
    
    # Use the existing collection that we updated with DL metadata
    reference_collection = "extended_contextualized_collection"
    
    # Test sentence
    test_sentence = "Fail fast is our core principle"
    
    print(f"🔍 Test sentence: '{test_sentence}'")
    print(f"📊 Using collection: {reference_collection}")
    
    try:
        # Check collection metadata
        print(f"\n📋 Checking DL metadata in collection...")
        validation = qdrant_service.validate_dl_metadata_completeness(reference_collection)
        print(f"   Total points: {validation['total_points']}")
        print(f"   Complete DL metadata: {validation['complete_metadata']}")
        print(f"   Completion rate: {validation['completion_rate']:.2%}")
        
        # Test DL-based queries
        print(f"\n🎯 Testing DL-based queries...")
        
        # Query by specific DL categories
        innovation_sentences = qdrant_service.query_by_dl_metadata(
            collection_name=reference_collection,
            dl_category="Innovation & Change",
            limit=5
        )
        print(f"   Innovation & Change: {len(innovation_sentences)} sentences")
        
        performance_sentences = qdrant_service.query_by_dl_metadata(
            collection_name=reference_collection,
            dl_category="Performance & Results",
            limit=5
        )
        print(f"   Performance & Results: {len(performance_sentences)} sentences")
        
        # Query by specific subcategory
        fail_fast_sentences = qdrant_service.query_by_dl_metadata(
            collection_name=reference_collection,
            dl_subcategory="Fail Fast, Learn Faster",
            limit=10
        )
        print(f"   Fail Fast, Learn Faster: {len(fail_fast_sentences)} sentences")
        
        if fail_fast_sentences:
            print(f"\n   📝 'Fail Fast, Learn Faster' sentences:")
            for i, sentence in enumerate(fail_fast_sentences, 1):
                phrase = sentence.actual_phrase or sentence.text
                if phrase.startswith("Domain Logic example phrase: "):
                    phrase = phrase[len("Domain Logic example phrase: "):]
                print(f"      {i}. {phrase}")
        
        # Run enhanced inference
        print(f"\n🧠 Running enhanced inference...")
        result = inference_engine.infer_dl_archetype(test_sentence, reference_collection)
        
        if result.primary_match:
            print(f"\n✅ ENHANCED INFERENCE RESULT:")
            print(f"🎯 Primary Match:")
            print(f"   Archetype: {result.primary_match.archetype}")
            print(f"   Similarity Score: {result.primary_match.similarity_score:.4f}")
            print(f"   Confidence Level: {result.primary_match.confidence_level.value}")
            
            # Try to extract DL metadata from similar matches
            print(f"\n🏷️  DL Metadata Analysis:")
            
            # Get the most similar sentences to analyze their DL labels
            similar_sentences = qdrant_service.search_similar(
                collection_name=reference_collection,
                query_vector=result.input_embedding,
                limit=10
            )
            
            if similar_sentences:
                dl_categories = {}
                dl_subcategories = {}
                
                for sentence in similar_sentences:
                    if hasattr(sentence, 'dl_category') and sentence.dl_category:
                        dl_categories[sentence.dl_category] = dl_categories.get(sentence.dl_category, 0) + 1
                    if hasattr(sentence, 'dl_subcategory') and sentence.dl_subcategory:
                        dl_subcategories[sentence.dl_subcategory] = dl_subcategories.get(sentence.dl_subcategory, 0) + 1
                
                if dl_categories:
                    print(f"   📊 Most similar DL categories:")
                    for category, count in sorted(dl_categories.items(), key=lambda x: x[1], reverse=True)[:3]:
                        print(f"      - {category}: {count} matches")
                
                if dl_subcategories:
                    print(f"   📊 Most similar DL subcategories:")
                    for subcategory, count in sorted(dl_subcategories.items(), key=lambda x: x[1], reverse=True)[:3]:
                        print(f"      - {subcategory}: {count} matches")
        
        else:
            print(f"❌ No primary match found")
        
        print(f"\n🎉 DL-Enhanced Inference Test Complete!")
        
    except Exception as e:
        print(f"❌ Error during DL-enhanced inference: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dl_enhanced_inference()
