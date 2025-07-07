#!/usr/bin/env python3
"""
Test the enhanced inference engine with real production data
"""
from app.core.inference_engine import EnhancedInferenceEngine
from app.services.qdrant_client import QdrantService
from app.services.openai_client import OpenAIService
import json

def test_real_inference():
    print("🚀 Testing Enhanced Inference Engine with Production Data")
    print("=" * 60)
    
    # Initialize services
    qdrant_service = QdrantService()
    openai_service = OpenAIService()
    inference_engine = EnhancedInferenceEngine(openai_service, qdrant_service)
    
    # Check available collections
    print("📋 Available collections:")
    try:
        collections_info = qdrant_service.client.get_collections()
        collections = [c.name for c in collections_info.collections]
        for collection in collections:
            print(f"  - {collection}")
    except Exception as e:
        print(f"Error listing collections: {e}")
        # Try to use a known collection name
        collections = ["extended_contextualized_collection", "collection3", "contextualized_collection"]
        print("  Using known collection names...")
        for collection in collections:
            if qdrant_service.collection_exists(collection):
                print(f"  - {collection} ✓")
            else:
                print(f"  - {collection} ✗")
    
    # Use the latest collection with the 600-entry contextualized dataset
    reference_collection = "extended_contextualized_collection"
    
    # Test sentence
    test_sentence = "Fail fast is our core principle"
    
    print(f"\n🔍 Classifying: '{test_sentence}'")
    print(f"📊 Using reference database: {reference_collection}")
    print("-" * 60)
    
    try:
        # Run enhanced inference
        result = inference_engine.infer_dl_archetype(test_sentence, reference_collection)
        
        print(f"\n✅ ENHANCED INFERENCE RESULT:")
        
        # Debug check
        if result.primary_match is None:
            print("❌ Primary match is None!")
            print(f"   Classification Status: {result.classification_status.value}")
            print(f"   Warnings: {result.warnings}")
            print(f"   Recommendations: {result.recommendations}")
            return
        
        print(f"🎯 Primary Match:")
        print(f"   Archetype: {result.primary_match.archetype}")
        print(f"   Similarity Score: {result.primary_match.similarity_score:.4f}")
        print(f"   Adjusted Confidence: {result.primary_match.adjusted_confidence:.4f}")
        print(f"   Confidence Level: {result.primary_match.confidence_level.value}")
        print(f"   Reliability Score: {result.primary_match.reliability_score:.4f}")
        print(f"   Cluster ID: {result.primary_match.cluster_id}")
        
        print(f"\n📊 Multi-Factor Confidence Assessment:")
        mfc = result.multi_factor_confidence
        print(f"   Base Similarity: {mfc.base_similarity:.4f}")
        print(f"   Percentile Rank: {mfc.percentile_rank:.4f}")
        print(f"   Gap to Second: {mfc.gap_to_second:.4f}")
        print(f"   Cluster Size Factor: {mfc.cluster_size_factor:.4f}")
        print(f"   Cluster Quality Factor: {mfc.cluster_quality_factor:.4f}")
        print(f"   Semantic Boost: {mfc.semantic_boost:.4f}")
        print(f"   Final Confidence: {mfc.final_confidence:.4f}")
        
        print(f"\n🧠 Semantic Analysis:")
        semantic = result.primary_match.semantic_analysis
        print(f"   Keyword Overlap Score: {semantic.keyword_overlap_score:.4f}")
        print(f"   Domain Terminology Present: {semantic.domain_terminology_present}")
        print(f"   Semantic Coherence Score: {semantic.semantic_coherence_score:.4f}")
        print(f"   Terminology Boost: {semantic.terminology_boost:.4f}")
        print(f"   Matched Keywords: {semantic.matched_keywords}")
        print(f"   Missing Keywords: {semantic.missing_keywords[:5]}...")  # Show first 5
        
        print(f"\n🏷️  Classification Status: {result.classification_status.value}")
        print(f"🤖 Model Compatibility: {result.model_compatibility_score:.4f}")
        print(f"🔍 Training Leakage Detected: {result.training_leakage_detected}")
        
        if result.alternative_matches:
            print(f"\n🔄 Top Alternative Matches:")
            for i, alt in enumerate(result.alternative_matches[:3], 1):
                print(f"   {i}. {alt.archetype}: {alt.similarity_score:.4f} ({alt.confidence_level.value})")
        
        if result.warnings:
            print(f"\n⚠️  Warnings:")
            for warning in result.warnings:
                print(f"   - {warning}")
                
        if result.recommendations:
            print(f"\n💡 Recommendations:")
            for rec in result.recommendations:
                print(f"   - {rec}")
        
        print(f"\n📝 Input Sentence: {result.input_sentence}")
        print(f"🔄 Contextualized: {result.contextualized_sentence}")
        print(f"⏰ Processed: {result.processing_timestamp}")
        
        # Save detailed result to file
        with open('inference_result.json', 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"\n💾 Detailed result saved to: inference_result.json")
        
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_real_inference()
