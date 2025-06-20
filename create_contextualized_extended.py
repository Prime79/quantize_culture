#!/usr/bin/env python3
"""
Create a contextualized version of the extended dataset (600 entries)
by adding "Domain Logic example phrase: " prefix to each sentence.
"""

import json
import sys
from pathlib import Path

def contextualize_extended_dataset():
    """Create contextualized version of extended_dl_sentences.json"""
    
    input_file = "extended_dl_sentences.json"
    output_file = "extended_dl_sentences_contextualized.json"
    
    print(f"📄 Loading original dataset: {input_file}")
    
    # Load original data
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: {input_file} not found!")
        return False
    except json.JSONDecodeError as e:
        print(f"❌ Error reading JSON: {e}")
        return False
    
    # Create contextualized version
    contextualized_data = {}
    total_sentences = 0
    
    print("🔄 Contextualizing sentences...")
    
    for category, subcategories in data.items():
        contextualized_data[category] = {}
        
        for subcategory, sentences in subcategories.items():
            contextualized_sentences = []
            
            for sentence in sentences:
                # Add domain context prefix
                contextualized_sentence = f"Domain Logic example phrase: {sentence}"
                contextualized_sentences.append(contextualized_sentence)
                total_sentences += 1
            
            contextualized_data[category][subcategory] = contextualized_sentences
        
        print(f"  ✅ Processed category: {category}")
    
    # Save contextualized data
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(contextualized_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Successfully created contextualized dataset!")
        print(f"📊 Total sentences contextualized: {total_sentences}")
        print(f"💾 Output saved to: {output_file}")
        
        # Verify file size
        output_path = Path(output_file)
        file_size = output_path.stat().st_size / 1024  # KB
        print(f"📁 File size: {file_size:.1f} KB")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving file: {e}")
        return False

def main():
    """Main execution function"""
    print("🚀 CREATING CONTEXTUALIZED EXTENDED DATASET")
    print("=" * 60)
    
    success = contextualize_extended_dataset()
    
    if success:
        print("\n🎉 CONTEXTUALIZATION COMPLETE!")
        print("\nNext steps:")
        print("1. Run full workflow on contextualized data:")
        print("   python3 -c \"from run_full_workflow import run_full_workflow; run_full_workflow('extended_dl_sentences_contextualized.json', 'extended_contextualized_collection')\"")
        print("2. Compare results with benchmark_comparison.py")
    else:
        print("\n❌ CONTEXTUALIZATION FAILED!")
        sys.exit(1)

if __name__ == "__main__":
    main()
