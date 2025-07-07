#!/usr/bin/env python3
"""
Investigate Specific Coordinate Issue

This script specifically investigates the coordinate mismatch issue reported:
- Coordinate (0.982, 1.395) should have "ideally you would form a team..."
- But user sees "and then getting to the real pain..."
- And the "ideally..." sentence appears at (8.351, 5.412) and (7.885, 4.666)
"""

import pandas as pd
import numpy as np
import json

def investigate_coordinate_issue():
    """Investigate the specific coordinate mismatch issue."""
    print("🔍 INVESTIGATING COORDINATE MISMATCH ISSUE")
    print("=" * 60)
    
    # Load the CSV data
    csv_path = "/Users/tamastorkoly/Documents/Doc-MB15/Projects/quantize_culture/umap_2d_data.csv"
    df = pd.read_csv(csv_path)
    
    print(f"📊 Loaded {len(df)} records from CSV")
    print(f"📈 Columns: {list(df.columns)}")
    
    # Define the coordinates mentioned by the user
    target_coords = [
        (0.982, 1.395),   # Should have "ideally you would form..."
        (8.351, 5.412),  # User says has "ideally you would form..."
        (7.885, 4.666)   # User says has "ideally you would form..."
    ]
    
    target_passages = [
        "ideally you would form a team",
        "and then getting to the real pain"
    ]
    
    print(f"\n🎯 INVESTIGATING SPECIFIC COORDINATES:")
    print("=" * 60)
    
    # Check what's actually at each coordinate
    for i, (x, y) in enumerate(target_coords):
        print(f"\n{i+1}. Checking coordinate ({x}, {y}):")
        
        # Find the closest match (allowing for small floating point differences)
        distances = np.sqrt((df['UMAP_1'] - x)**2 + (df['UMAP_2'] - y)**2)
        closest_idx = distances.idxmin()
        closest_distance = distances.min()
        
        closest_row = df.iloc[closest_idx]
        
        print(f"   📍 Exact coordinates: ({closest_row['UMAP_1']:.6f}, {closest_row['UMAP_2']:.6f})")
        print(f"   📏 Distance from target: {closest_distance:.6f}")
        print(f"   🏷️  Label: {closest_row['Dominant_Logic']}")
        print(f"   📝 Passage: \"{closest_row['Passage']}\"")
        
        if closest_distance > 0.001:
            print(f"   ⚠️  No exact match found! Closest is {closest_distance:.6f} units away")
    
    print(f"\n🔍 SEARCHING FOR TARGET PASSAGES:")
    print("=" * 60)
    
    # Search for the target passages
    for passage_fragment in target_passages:
        print(f"\n🔎 Searching for passages containing: \"{passage_fragment}\"")
        
        matches = df[df['Passage'].str.contains(passage_fragment, case=False, na=False)]
        
        if len(matches) == 0:
            print(f"   ❌ No matches found!")
        else:
            print(f"   ✅ Found {len(matches)} match(es):")
            for idx, row in matches.iterrows():
                print(f"   📍 Coordinates: ({row['UMAP_1']:.6f}, {row['UMAP_2']:.6f})")
                print(f"   🏷️  Label: {row['Dominant_Logic']}")
                print(f"   📝 Full passage: \"{row['Passage']}\"")
                print()
    
    print(f"\n🔍 CHECKING FOR DATA CORRUPTION PATTERNS:")
    print("=" * 60)
    
    # Check if there are any duplicate coordinates
    coord_duplicates = df.groupby(['UMAP_1', 'UMAP_2']).size()
    duplicates = coord_duplicates[coord_duplicates > 1]
    
    if len(duplicates) > 0:
        print(f"❌ Found {len(duplicates)} coordinate duplicates!")
        for (x, y), count in duplicates.items():
            print(f"   📍 ({x}, {y}): {count} occurrences")
            dup_rows = df[(df['UMAP_1'] == x) & (df['UMAP_2'] == y)]
            for idx, row in dup_rows.iterrows():
                print(f"     - \"{row['Passage'][:50]}...\" ({row['Dominant_Logic']})")
    else:
        print(f"✅ No coordinate duplicates found in CSV data")
    
    # Check if there are any passage duplicates
    passage_duplicates = df.groupby('Passage').size()
    dup_passages = passage_duplicates[passage_duplicates > 1]
    
    if len(dup_passages) > 0:
        print(f"\n❌ Found {len(dup_passages)} passage duplicates!")
        for passage, count in dup_passages.items():
            print(f"   📝 \"{passage[:50]}...\": {count} occurrences")
            dup_rows = df[df['Passage'] == passage]
            for idx, row in dup_rows.iterrows():
                print(f"     - Coordinates: ({row['UMAP_1']:.6f}, {row['UMAP_2']:.6f}) - {row['Dominant_Logic']}")
    else:
        print(f"✅ No passage duplicates found in CSV data")
    
    # Export detailed data for manual inspection
    print(f"\n💾 EXPORTING DETAILED DATA:")
    print("=" * 60)
    
    # Export all data with row indices for debugging
    debug_df = df.copy()
    debug_df['Row_Index'] = debug_df.index
    debug_df.to_csv('debug_umap_data_with_indices.csv', index=False)
    print(f"✅ Exported debug data to: debug_umap_data_with_indices.csv")
    
    # Export specific coordinate neighborhoods
    for i, (x, y) in enumerate(target_coords):
        # Find all points within 0.1 units of target coordinate
        mask = (np.abs(df['UMAP_1'] - x) < 0.1) & (np.abs(df['UMAP_2'] - y) < 0.1)
        neighborhood = df[mask].copy()
        
        if len(neighborhood) > 0:
            filename = f'coordinate_neighborhood_{i+1}_{x}_{y}.csv'
            neighborhood.to_csv(filename, index=False)
            print(f"✅ Exported neighborhood {i+1} to: {filename}")
        else:
            print(f"⚠️  No points found near coordinate ({x}, {y})")
    
    return df

def check_plotly_data_generation():
    """Check if the issue occurs during Plotly data generation."""
    print(f"\n🔍 CHECKING PLOTLY DATA GENERATION:")
    print("=" * 60)
    
    # Load the original CSV
    csv_path = "/Users/tamastorkoly/Documents/Doc-MB15/Projects/quantize_culture/umap_2d_data.csv"
    df = pd.read_csv(csv_path)
    
    # Simulate the exact steps from create_interactive_umap_plot.py
    print(f"📊 Original data shape: {df.shape}")
    
    # Create wrapped text (as done in the plotting script)
    def wrap_text_for_hover(text, max_line_length=80):
        import textwrap
        lines = text.split('\n')
        wrapped_lines = []
        for line in lines:
            if len(line) <= max_line_length:
                wrapped_lines.append(line)
            else:
                wrapped_lines.extend(textwrap.wrap(line, width=max_line_length))
        return '<br>'.join(wrapped_lines)
    
    df['Passage_Wrapped'] = df['Passage'].apply(lambda x: wrap_text_for_hover(x, 80))
    
    print(f"📊 After text wrapping: {df.shape}")
    
    # Check if text wrapping caused any issues
    print(f"\n🔍 Checking text wrapping effects:")
    
    # Find the "ideally" passage
    ideally_mask = df['Passage'].str.contains('ideally you would form', case=False, na=False)
    ideally_rows = df[ideally_mask]
    
    if len(ideally_rows) > 0:
        print(f"✅ Found 'ideally' passage:")
        for idx, row in ideally_rows.iterrows():
            print(f"   📍 Coordinates: ({row['UMAP_1']:.6f}, {row['UMAP_2']:.6f})")
            print(f"   📝 Original: \"{row['Passage'][:60]}...\"")
            print(f"   📝 Wrapped: \"{row['Passage_Wrapped'][:60]}...\"")
    
    # Find the "pain point" passage
    pain_mask = df['Passage'].str.contains('real pain point', case=False, na=False)
    pain_rows = df[pain_mask]
    
    if len(pain_rows) > 0:
        print(f"\n✅ Found 'pain point' passage:")
        for idx, row in pain_rows.iterrows():
            print(f"   📍 Coordinates: ({row['UMAP_1']:.6f}, {row['UMAP_2']:.6f})")
            print(f"   📝 Original: \"{row['Passage'][:60]}...\"")
            print(f"   📝 Wrapped: \"{row['Passage_Wrapped'][:60]}...\"")
    
    # Export the processed data that would go to Plotly
    plotly_data = df[['UMAP_1', 'UMAP_2', 'Dominant_Logic', 'Passage_Wrapped']].copy()
    plotly_data.to_csv('debug_plotly_input_data.csv', index=False)
    print(f"\n💾 Exported Plotly input data to: debug_plotly_input_data.csv")
    
    return df

def main():
    """Main investigation function."""
    print("🚨 COORDINATE MISMATCH INVESTIGATION")
    print("="*70)
    
    # Step 1: Investigate the coordinate issue
    df = investigate_coordinate_issue()
    
    # Step 2: Check Plotly data generation
    check_plotly_data_generation()
    
    print(f"\n🎯 SUMMARY:")
    print(f"✅ Investigation complete - debug files generated")
    print(f"✅ Check the exported CSV files for detailed analysis")
    print(f"✅ If data in CSV is correct, the issue is in the visualization layer")
    
    print(f"\n📁 Debug files created:")
    print(f"   • debug_umap_data_with_indices.csv")
    print(f"   • debug_plotly_input_data.csv") 
    print(f"   • coordinate_neighborhood_*.csv (if any)")

if __name__ == "__main__":
    main()
