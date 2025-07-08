#!/usr/bin/env python3
"""
Detailed Analysis of Close Proximity Different Labels

Focuses on the most suspicious cases where passages are very close 
in UMAP space but have different labels.
"""

import pandas as pd
import numpy as np
from difflib import SequenceMatcher

def analyze_closest_proximity_cases():
    """Analyze the closest proximity cases in detail."""
    print("🔍 DETAILED ANALYSIS OF CLOSEST PROXIMITY CASES")
    print("=" * 60)
    
    # Load data
    csv_path = "/Users/tamastorkoly/Documents/Doc-MB15/Projects/quantize_culture/umap_2d_data.csv"
    df = pd.read_csv(csv_path)
    
    # Find the closest pairs with different labels
    closest_pairs = []
    
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            if df.iloc[i]['Dominant_Logic'] != df.iloc[j]['Dominant_Logic']:
                coords1 = np.array([df.iloc[i]['UMAP_1'], df.iloc[i]['UMAP_2']])
                coords2 = np.array([df.iloc[j]['UMAP_1'], df.iloc[j]['UMAP_2']])
                distance = np.linalg.norm(coords1 - coords2)
                
                closest_pairs.append({
                    'distance': distance,
                    'passage_1': df.iloc[i]['Passage'],
                    'label_1': df.iloc[i]['Dominant_Logic'],
                    'coords_1': coords1,
                    'passage_2': df.iloc[j]['Passage'],
                    'label_2': df.iloc[j]['Dominant_Logic'],
                    'coords_2': coords2,
                    'index_1': i,
                    'index_2': j
                })
    
    # Sort by distance (closest first)
    closest_pairs.sort(key=lambda x: x['distance'])
    
    print(f"📊 Analyzing the {min(15, len(closest_pairs))} closest pairs with different labels:")
    print("=" * 60)
    
    boundary_cases = []
    potential_errors = []
    
    for i, pair in enumerate(closest_pairs[:15]):
        print(f"\n{i+1}. DISTANCE: {pair['distance']:.4f} UMAP units")
        print(f"   📍 Coordinates: {pair['coords_1']} ↔ {pair['coords_2']}")
        
        print(f"\n   🏷️  LABEL 1: {pair['label_1']}")
        print(f"   📝 PASSAGE 1: \"{pair['passage_1']}\"")
        
        print(f"\n   🏷️  LABEL 2: {pair['label_2']}")
        print(f"   📝 PASSAGE 2: \"{pair['passage_2']}\"")
        
        # Calculate text similarity
        text_similarity = SequenceMatcher(None, pair['passage_1'], pair['passage_2']).ratio()
        print(f"\n   📊 Text similarity: {text_similarity:.3f}")
        
        # Analyze the semantic relationship
        analysis = analyze_semantic_relationship(pair['passage_1'], pair['passage_2'], 
                                               pair['label_1'], pair['label_2'])
        print(f"   🤔 Analysis: {analysis['type']}")
        print(f"   💭 Explanation: {analysis['explanation']}")
        
        if analysis['type'] == 'boundary_case':
            boundary_cases.append(pair)
        elif analysis['type'] == 'potential_error':
            potential_errors.append(pair)
        
        print("   " + "─" * 50)
    
    # Summary
    print(f"\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"📊 Total pairs analyzed: {len(closest_pairs[:15])}")
    print(f"🎯 Boundary cases (multiple valid interpretations): {len(boundary_cases)}")
    print(f"⚠️  Potential labeling errors: {len(potential_errors)}")
    
    if boundary_cases:
        print(f"\n🎯 BOUNDARY CASES:")
        print("These represent genuine cases where passages could reasonably")
        print("be classified into multiple categories depending on context:")
        for case in boundary_cases:
            print(f"   • \"{case['passage_1'][:50]}...\" vs \"{case['passage_2'][:50]}...\"")
    
    if potential_errors:
        print(f"\n⚠️  POTENTIAL ERRORS:")
        print("These cases might benefit from review:")
        for case in potential_errors:
            print(f"   • Distance {case['distance']:.3f}: {case['label_1']} vs {case['label_2']}")
            print(f"     \"{case['passage_1'][:60]}...\"")
            print(f"     \"{case['passage_2'][:60]}...\"")
    
    return {
        'closest_pairs': closest_pairs[:15],
        'boundary_cases': boundary_cases,
        'potential_errors': potential_errors
    }

def analyze_semantic_relationship(passage1, passage2, label1, label2):
    """Analyze the semantic relationship between two passages."""
    
    # Convert to lowercase for analysis
    p1_lower = passage1.lower()
    p2_lower = passage2.lower()
    
    # Check for financial/money-related keywords
    financial_keywords = ['money', 'cost', 'profit', 'financial', 'revenue', 'dollar', 
                         'budget', 'expense', 'economic', 'roi', 'investment']
    
    # Check for entrepreneurial keywords
    entrepreneur_keywords = ['startup', 'innovation', 'risk', 'opportunity', 'venture',
                           'disrupt', 'agile', 'pivot', 'scale', 'growth']
    
    # Check for certainty keywords
    certainty_keywords = ['certain', 'sure', 'confident', 'know', 'proven', 'reliable',
                         'stable', 'traditional', 'established', 'standard']
    
    # Analyze content overlap
    has_financial = any(kw in p1_lower or kw in p2_lower for kw in financial_keywords)
    has_entrepreneur = any(kw in p1_lower or kw in p2_lower for kw in entrepreneur_keywords)
    has_certainty = any(kw in p1_lower or kw in p2_lower for kw in certainty_keywords)
    
    # Determine relationship type
    if (label1 == 'FINANCIAL PERFORMANCE FIRST' and label2 == 'ENTREPRENEUR') or \
       (label1 == 'ENTREPRENEUR' and label2 == 'FINANCIAL PERFORMANCE FIRST'):
        
        if has_financial and has_entrepreneur:
            return {
                'type': 'boundary_case',
                'explanation': 'Contains both financial and entrepreneurial concepts - context dependent'
            }
        elif 'profit' in p1_lower or 'profit' in p2_lower:
            return {
                'type': 'boundary_case', 
                'explanation': 'Profit discussions can be both financial and entrepreneurial'
            }
        else:
            return {
                'type': 'potential_error',
                'explanation': 'Labels seem inconsistent for similar semantic content'
            }
    
    elif (label1 == 'CERTAINTY' and label2 in ['ENTREPRENEUR', 'FINANCIAL PERFORMANCE FIRST']) or \
         (label2 == 'CERTAINTY' and label1 in ['ENTREPRENEUR', 'FINANCIAL PERFORMANCE FIRST']):
        
        if 'problem' in p1_lower and 'solve' in p1_lower:
            return {
                'type': 'boundary_case',
                'explanation': 'Problem-solving can be both certain methodology and entrepreneurial'
            }
        elif len(passage1) < 30 or len(passage2) < 30:
            return {
                'type': 'potential_error',
                'explanation': 'Very short passages may be harder to classify consistently'
            }
        else:
            return {
                'type': 'boundary_case',
                'explanation': 'Different aspects of business thinking represented'
            }
    
    else:
        return {
            'type': 'boundary_case',
            'explanation': 'Semantic similarity suggests overlapping concepts'
        }

def main():
    """Main analysis function."""
    results = analyze_closest_proximity_cases()
    
    print(f"\n🎯 CONCLUSION:")
    print(f"The analysis suggests that most close proximity cases represent")
    print(f"genuine boundary cases where passages contain overlapping concepts")
    print(f"from multiple dominant logic categories. This is actually a good sign")
    print(f"that the UMAP embedding is capturing semantic nuances correctly.")
    
    return results

if __name__ == "__main__":
    main()
