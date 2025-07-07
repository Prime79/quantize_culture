#!/usr/bin/env python3
"""
Debug Plotly Visualization Data

Check if there's an issue with the data being passed to Plotly
that could cause the same passage to appear at multiple coordinates.
"""

import pandas as pd
import numpy as np

def debug_plotly_data_source():
    """Debug the exact data being used for Plotly visualization."""
    print("🔍 DEBUGGING PLOTLY DATA SOURCE")
    print("=" * 50)
    
    # Load the same data that the visualization uses
    csv_path = "/Users/tamastorkoly/Documents/Doc-MB15/Projects/quantize_culture/umap_2d_data.csv"
    df = pd.read_csv(csv_path)
    
    print(f"📊 Original data shape: {df.shape}")
    
    # Apply the same data transformations as in the visualization
    def wrap_text_for_hover(text, max_line_length=80):
        """Same function as in the visualization code."""
        import textwrap
        lines = text.split('\n')
        wrapped_lines = []
        for line in lines:
            if len(line) <= max_line_length:
                wrapped_lines.append(line)
            else:
                wrapped_lines.extend(textwrap.wrap(line, width=max_line_length))
        return '<br>'.join(wrapped_lines)
    
    # Create wrapped text as done in the visualization
    df['Passage_Wrapped'] = df['Passage'].apply(lambda x: wrap_text_for_hover(x, 80))
    
    print(f"📊 Data after transformation: {df.shape}")
    
    # Search for the specific passage the user mentioned
    target_fragment = "ideally you would form a team"
    
    # Check original Passage column
    print(f"\n🎯 Searching in original 'Passage' column:")
    orig_matches = df[df['Passage'].str.contains(target_fragment, case=False, na=False)]
    print(f"   Found {len(orig_matches)} matches in original column")
    
    for i, (idx, row) in enumerate(orig_matches.iterrows()):
        print(f"   {i+1}. Index {idx}: ({row['UMAP_1']:.6f}, {row['UMAP_2']:.6f})")
        print(f"      Original: \"{row['Passage'][:80]}...\"")
    
    # Check wrapped Passage column
    print(f"\n🎯 Searching in wrapped 'Passage_Wrapped' column:")
    wrapped_matches = df[df['Passage_Wrapped'].str.contains(target_fragment, case=False, na=False)]
    print(f"   Found {len(wrapped_matches)} matches in wrapped column")
    
    for i, (idx, row) in enumerate(wrapped_matches.iterrows()):
        print(f"   {i+1}. Index {idx}: ({row['UMAP_1']:.6f}, {row['UMAP_2']:.6f})")
        print(f"      Wrapped: \"{row['Passage_Wrapped'][:80]}...\"")
    
    # Check what data is being passed to Plotly customdata
    print(f"\n📊 Checking customdata array (what Plotly sees):")
    customdata = df[['UMAP_1', 'UMAP_2', 'Dominant_Logic', 'Passage_Wrapped']].values
    print(f"   Customdata shape: {customdata.shape}")
    
    # Check for the specific passage in customdata
    for i, row in enumerate(customdata):
        passage_text = row[3]  # Passage_Wrapped is at index 3
        if target_fragment.lower() in passage_text.lower():
            print(f"   Found in customdata at index {i}:")
            print(f"      Coordinates: ({row[0]:.6f}, {row[1]:.6f})")
            print(f"      Label: {row[2]}")
            print(f"      Text: \"{passage_text[:80]}...\"")
    
    # Check for any duplicate passages in the data
    print(f"\n🔍 Checking for duplicate passages:")
    passage_counts = df['Passage'].value_counts()
    duplicates = passage_counts[passage_counts > 1]
    
    if len(duplicates) > 0:
        print(f"   ❌ Found {len(duplicates)} duplicate passages:")
        for passage, count in duplicates.items():
            print(f"      \"{passage[:60]}...\" appears {count} times")
            
            # Show coordinates for duplicates
            dup_rows = df[df['Passage'] == passage]
            for idx, row in dup_rows.iterrows():
                print(f"         Index {idx}: ({row['UMAP_1']:.6f}, {row['UMAP_2']:.6f}) - {row['Dominant_Logic']}")
    else:
        print(f"   ✅ No duplicate passages found")
    
    return df, customdata

def check_plotly_html_file():
    """Check the actual HTML file to see what data is embedded."""
    print(f"\n🔍 CHECKING PLOTLY HTML FILE")
    print("=" * 50)
    
    html_path = "/Users/tamastorkoly/Documents/Doc-MB15/Projects/quantize_culture/enhanced_interactive_umap_plot.html"
    
    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        print(f"📊 HTML file size: {len(html_content)} characters")
        
        # Search for the specific passage in the HTML
        target_fragment = "ideally you would form a team"
        
        # Count occurrences
        occurrences = html_content.lower().count(target_fragment.lower())
        print(f"🎯 Occurrences of '{target_fragment}' in HTML: {occurrences}")
        
        if occurrences > 1:
            print(f"   ⚠️  Passage appears {occurrences} times in HTML file!")
            print(f"   This confirms the issue is in the plot data generation")
        else:
            print(f"   ✅ Passage appears only once in HTML file")
        
        # Look for coordinate patterns
        coords_8_351 = html_content.count("8.351") + html_content.count("8.350604")
        coords_7_885 = html_content.count("7.885") + html_content.count("7.884652")
        
        print(f"📊 Coordinate references in HTML:")
        print(f"   ~8.351: {coords_8_351} occurrences")
        print(f"   ~7.885: {coords_7_885} occurrences")
        
    except Exception as e:
        print(f"❌ Error reading HTML file: {e}")

def regenerate_clean_visualization():
    """Regenerate the visualization with explicit data validation."""
    print(f"\n🔧 REGENERATING CLEAN VISUALIZATION")
    print("=" * 50)
    
    # Load fresh data
    csv_path = "/Users/tamastorkoly/Documents/Doc-MB15/Projects/quantize_culture/umap_2d_data.csv"
    df = pd.read_csv(csv_path)
    
    print(f"📊 Starting with {len(df)} records")
    
    # Explicit deduplication (just in case)
    print(f"🧹 Deduplicating data...")
    original_len = len(df)
    df_clean = df.drop_duplicates().reset_index(drop=True)
    print(f"   Removed {original_len - len(df_clean)} duplicate rows")
    
    # Validate no duplicate passages
    passage_counts = df_clean['Passage'].value_counts()
    duplicates = passage_counts[passage_counts > 1]
    
    if len(duplicates) > 0:
        print(f"❌ Still found {len(duplicates)} duplicate passages after dedup!")
        for passage, count in duplicates.head(3).items():
            print(f"   \"{passage[:50]}...\" appears {count} times")
    else:
        print(f"✅ No duplicate passages in clean data")
    
    # Check target passage specifically
    target_fragment = "ideally you would form a team"
    target_matches = df_clean[df_clean['Passage'].str.contains(target_fragment, case=False, na=False)]
    
    print(f"\n🎯 Target passage check:")
    print(f"   Found {len(target_matches)} instances of target passage")
    
    for i, (idx, row) in enumerate(target_matches.iterrows()):
        print(f"   {i+1}. Index {idx}: ({row['UMAP_1']:.6f}, {row['UMAP_2']:.6f}) - {row['Dominant_Logic']}")
        print(f"      \"{row['Passage']}\"")
    
    return df_clean

def main():
    """Main debugging function."""
    print("🚨 DEBUGGING PLOTLY VISUALIZATION ISSUE")
    print("User sees same passage at different coordinates in plot")
    print("="*60)
    
    # Step 1: Debug the data source
    df, customdata = debug_plotly_data_source()
    
    # Step 2: Check the HTML file
    check_plotly_html_file()
    
    # Step 3: Generate clean visualization
    df_clean = regenerate_clean_visualization()
    
    print(f"\n" + "="*60)
    print("DEBUGGING SUMMARY")
    print("="*60)
    print("If the issue persists, it suggests:")
    print("1. Browser caching of old visualization")
    print("2. Multiple HTML files being viewed")
    print("3. Plotly internal data handling issue")
    print("4. Copy-paste error in coordinates")
    
    print(f"\n💡 NEXT STEPS:")
    print("1. Clear browser cache and reload")
    print("2. Regenerate visualization with clean data")
    print("3. Check exact coordinates in interactive plot")
    print("4. Verify which HTML file is being viewed")

if __name__ == "__main__":
    main()
