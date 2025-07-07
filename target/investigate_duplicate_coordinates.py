#!/usr/bin/env python3
"""
Investigate Specific Duplicate Issue

User reported seeing the same passage "ideally you would form a team..." 
at two different UMAP coordinates:
- UMAP1: 8.351, UMAP2: 5.412
- UMAP1: 7.885, UMAP2: 4.666

This script will investigate this specific case and check for similar issues.
"""

import pandas as pd
import numpy as np

def investigate_specific_duplicate():
    """Investigate the specific duplicate case reported by user."""
    print("🔍 INVESTIGATING SPECIFIC DUPLICATE CASE")
    print("=" * 60)
    
    # Load the data
    csv_path = "/Users/tamastorkoly/Documents/Doc-MB15/Projects/quantize_culture/umap_2d_data.csv"
    df = pd.read_csv(csv_path)
    
    print(f"📊 Loaded {len(df)} total records")
    
    # Search for the specific passage mentioned
    target_passage_fragment = "ideally you would form a team"
    
    print(f"\n🎯 Searching for passages containing: '{target_passage_fragment}'")
    
    matching_rows = df[df['Passage'].str.contains(target_passage_fragment, case=False, na=False)]
    
    print(f"✅ Found {len(matching_rows)} matching records:")
    
    if len(matching_rows) > 0:
        print("\n📋 DETAILED ANALYSIS:")
        for i, (idx, row) in enumerate(matching_rows.iterrows()):
            print(f"\n{i+1}. Record Index: {idx}")
            print(f"   UMAP Coordinates: ({row['UMAP_1']:.6f}, {row['UMAP_2']:.6f})")
            print(f"   Dominant Logic: {row['Dominant_Logic']}")
            print(f"   Full Passage: \"{row['Passage']}\"")
            print(f"   Passage Length: {len(row['Passage'])} characters")
    
    # Check the specific coordinates mentioned by user
    coord1 = (8.351, 5.412)
    coord2 = (7.885, 4.666)
    
    print(f"\n🎯 Checking for records near the reported coordinates:")
    print(f"   Target 1: UMAP1={coord1[0]:.3f}, UMAP2={coord1[1]:.3f}")
    print(f"   Target 2: UMAP1={coord2[0]:.3f}, UMAP2={coord2[1]:.3f}")
    
    tolerance = 0.1  # Check within 0.1 units
    
    for coord_name, coord in [("Coordinate 1", coord1), ("Coordinate 2", coord2)]:
        nearby_rows = df[
            (abs(df['UMAP_1'] - coord[0]) < tolerance) & 
            (abs(df['UMAP_2'] - coord[1]) < tolerance)
        ]
        
        print(f"\n📍 Records near {coord_name}:")
        if len(nearby_rows) > 0:
            for i, (idx, row) in enumerate(nearby_rows.iterrows()):
                print(f"   {i+1}. Index {idx}: ({row['UMAP_1']:.6f}, {row['UMAP_2']:.6f})")
                print(f"      Label: {row['Dominant_Logic']}")
                print(f"      Passage: \"{row['Passage'][:80]}...\"")
        else:
            print(f"   ❌ No records found near {coord_name}")
    
    return matching_rows

def check_for_coordinate_duplicates():
    """Check for any other passages that appear at multiple UMAP coordinates."""
    print(f"\n" + "="*60)
    print("COMPREHENSIVE COORDINATE DUPLICATE CHECK")
    print("="*60)
    
    csv_path = "/Users/tamastorkoly/Documents/Doc-MB15/Projects/quantize_culture/umap_2d_data.csv"
    df = pd.read_csv(csv_path)
    
    # Group by passage and check for multiple coordinates
    passage_groups = df.groupby('Passage')
    
    coordinate_duplicates = []
    
    print(f"\n🔍 Checking all {len(passage_groups)} unique passages...")
    
    for passage, group in passage_groups:
        if len(group) > 1:
            # Check if coordinates are different
            coordinates = group[['UMAP_1', 'UMAP_2']].values
            labels = group['Dominant_Logic'].values
            
            # Check if any coordinates are different
            unique_coords = np.unique(coordinates, axis=0)
            
            if len(unique_coords) > 1:
                coordinate_duplicates.append({
                    'passage': passage,
                    'count': len(group),
                    'coordinates': coordinates.tolist(),
                    'labels': labels.tolist(),
                    'indices': group.index.tolist()
                })
    
    print(f"\n🎯 Found {len(coordinate_duplicates)} passages with multiple UMAP coordinates!")
    
    if coordinate_duplicates:
        print(f"\n❗ PASSAGES WITH MULTIPLE COORDINATES:")
        
        for i, dup in enumerate(coordinate_duplicates):
            print(f"\n{i+1}. Passage: \"{dup['passage'][:100]}...\"")
            print(f"   Occurrences: {dup['count']}")
            print(f"   Record Indices: {dup['indices']}")
            print(f"   Coordinates:")
            for j, (coord, label) in enumerate(zip(dup['coordinates'], dup['labels'])):
                print(f"     {j+1}. ({coord[0]:.6f}, {coord[1]:.6f}) - Label: {label}")
            
            # Calculate distance between coordinates
            if len(dup['coordinates']) == 2:
                coord1 = np.array(dup['coordinates'][0])
                coord2 = np.array(dup['coordinates'][1])
                distance = np.linalg.norm(coord1 - coord2)
                print(f"   Distance between coordinates: {distance:.6f} UMAP units")
    
    return coordinate_duplicates

def check_data_pipeline_integrity():
    """Check if the issue might be in the data pipeline."""
    print(f"\n" + "="*60)
    print("DATA PIPELINE INTEGRITY CHECK")
    print("="*60)
    
    csv_path = "/Users/tamastorkoly/Documents/Doc-MB15/Projects/quantize_culture/umap_2d_data.csv"
    df = pd.read_csv(csv_path)
    
    print(f"📊 Data file: {csv_path}")
    print(f"📊 Total records: {len(df)}")
    print(f"📊 Columns: {list(df.columns)}")
    
    # Check for any NaN values
    print(f"\n🔍 Checking for missing values:")
    missing_values = df.isnull().sum()
    for col, missing in missing_values.items():
        if missing > 0:
            print(f"   ❌ {col}: {missing} missing values")
        else:
            print(f"   ✅ {col}: No missing values")
    
    # Check for duplicate indices
    print(f"\n🔍 Checking for duplicate row indices:")
    duplicate_indices = df.index.duplicated().sum()
    if duplicate_indices > 0:
        print(f"   ❌ Found {duplicate_indices} duplicate indices")
    else:
        print(f"   ✅ No duplicate indices found")
    
    # Check coordinate ranges
    print(f"\n📊 UMAP coordinate ranges:")
    print(f"   UMAP_1: {df['UMAP_1'].min():.6f} to {df['UMAP_1'].max():.6f}")
    print(f"   UMAP_2: {df['UMAP_2'].min():.6f} to {df['UMAP_2'].max():.6f}")
    
    # Check for exact coordinate duplicates
    print(f"\n🔍 Checking for identical coordinates:")
    coordinate_pairs = df[['UMAP_1', 'UMAP_2']].round(6)  # Round to avoid floating point issues
    duplicate_coords = coordinate_pairs.duplicated().sum()
    
    if duplicate_coords > 0:
        print(f"   ⚠️  Found {duplicate_coords} records with identical coordinates")
        
        # Show examples
        duplicate_coord_mask = coordinate_pairs.duplicated(keep=False)
        duplicate_records = df[duplicate_coord_mask].sort_values(['UMAP_1', 'UMAP_2'])
        
        print(f"   📋 Examples of identical coordinates:")
        for i, (idx, row) in enumerate(duplicate_records.head(6).iterrows()):
            print(f"     {i+1}. Index {idx}: ({row['UMAP_1']:.6f}, {row['UMAP_2']:.6f}) - \"{row['Passage'][:50]}...\"")
    else:
        print(f"   ✅ No identical coordinates found")

def main():
    """Main investigation function."""
    print("🚨 INVESTIGATING REPORTED DUPLICATE COORDINATE ISSUE")
    print("User reported seeing the same passage at different coordinates:")
    print("- UMAP1: 8.351, UMAP2: 5.412")  
    print("- UMAP1: 7.885, UMAP2: 4.666")
    print("- Passage: 'ideally you would form a team...'")
    print("="*70)
    
    # Step 1: Investigate the specific case
    matching_rows = investigate_specific_duplicate()
    
    # Step 2: Check for other coordinate duplicates
    coordinate_duplicates = check_for_coordinate_duplicates()
    
    # Step 3: Check data pipeline integrity
    check_data_pipeline_integrity()
    
    # Summary
    print(f"\n" + "="*70)
    print("INVESTIGATION SUMMARY")
    print("="*70)
    
    if len(matching_rows) > 1:
        print(f"🚨 CONFIRMED: Multiple records found for the target passage")
        print(f"   Records found: {len(matching_rows)}")
    else:
        print(f"✅ Only one record found in CSV for the target passage")
    
    if coordinate_duplicates:
        print(f"🚨 ISSUE CONFIRMED: {len(coordinate_duplicates)} passages have multiple coordinates")
        print(f"   This indicates a problem in the data processing pipeline")
    else:
        print(f"✅ No other coordinate duplicates found")
    
    print(f"\n💡 RECOMMENDATIONS:")
    if coordinate_duplicates:
        print(f"   1. Check the UMAP processing pipeline for duplicate embedding creation")
        print(f"   2. Verify Qdrant data integrity")
        print(f"   3. Review the data extraction process")
        print(f"   4. Consider deduplicating the dataset")
    else:
        print(f"   1. The issue might be in the visualization code")
        print(f"   2. Check if the same passage appears in the plot generation")
        print(f"   3. Verify the Plotly data source")
    
    return {
        'matching_rows': matching_rows,
        'coordinate_duplicates': coordinate_duplicates
    }

if __name__ == "__main__":
    main()
