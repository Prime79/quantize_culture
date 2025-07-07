#!/usr/bin/env python3
"""
Quick test of the DLDataLoader implementation
"""
from app.data.loader import DLDataLoader
import json

def test_dl_data_loader():
    print("🧪 Testing DLDataLoader Implementation")
    print("=" * 50)
    
    # Test with sample data
    test_data = {
        "Performance & Results": {
            "Results Over Process": [
                "Just hit the number; we'll fix the paperwork later.",
                "End results trump following every guideline."
            ],
            "Play to Win": [
                "We're here to crush the competition, not co-exist.",
                "Second place is the first loser."
            ]
        },
        "Innovation & Change": {
            "Fail Fast, Learn Faster": [
                "Fail fast is our core principle.",
                "Each failure shortens the path to success."
            ]
        }
    }
    
    loader = DLDataLoader()
    
    # Test 1: Load DL structure
    print("🔍 Test 1: Loading DL structure")
    structure = loader.load_dl_structure(test_data)
    print(f"   ✅ Categories: {structure.total_categories}")
    print(f"   ✅ Subcategories: {structure.total_subcategories}")
    print(f"   ✅ Total sentences: {structure.total_sentences}")
    
    # Test 2: Extract sentences with metadata
    print("\n🔍 Test 2: Extracting sentences with DL metadata")
    sentences = loader.extract_sentences_with_dl_metadata(test_data)
    print(f"   ✅ Extracted {len(sentences)} sentences")
    
    for i, sentence in enumerate(sentences[:2], 1):
        print(f"   📝 Sentence {i}:")
        print(f"      Text: {sentence['text']}")
        print(f"      Category: {sentence['dl_category']}")
        print(f"      Subcategory: {sentence['dl_subcategory']}")
        print(f"      Archetype: {sentence['dl_archetype']}")
    
    # Test 3: Validate completeness
    print("\n🔍 Test 3: Validating DL completeness")
    validation = loader.validate_dl_completeness(sentences)
    print(f"   ✅ Valid: {validation.is_valid}")
    print(f"   ✅ Completion rate: {validation.completion_rate:.2%}")
    print(f"   ✅ Valid sentences: {validation.valid_sentences}/{validation.total_sentences}")
    
    # Test 4: Load from actual file
    print("\n🔍 Test 4: Loading from extended_dl_sentences.json")
    try:
        real_structure = loader.load_from_json_file('extended_dl_sentences.json')
        print(f"   ✅ Real data categories: {real_structure.total_categories}")
        print(f"   ✅ Real data subcategories: {real_structure.total_subcategories}")
        print(f"   ✅ Real data sentences: {real_structure.total_sentences}")
        
        # Sample some real sentences
        real_sentences = loader.extract_sentences_with_dl_metadata(real_structure.categories)
        print(f"   ✅ Extracted {len(real_sentences)} real sentences")
        
        # Show some examples
        print("\n   📝 Sample real sentences:")
        for i, sentence in enumerate(real_sentences[:3], 1):
            print(f"      {i}. {sentence['text'][:60]}...")
            print(f"         Category: {sentence['dl_category']}")
            print(f"         Subcategory: {sentence['dl_subcategory']}")
        
    except Exception as e:
        print(f"   ❌ Error loading real data: {e}")
    
    print("\n🎉 DLDataLoader test completed!")

if __name__ == "__main__":
    test_dl_data_loader()
