#!/usr/bin/env python3
"""
Interactive UMAP Demo Script

Quick demonstration of the key features and insights from the interactive 
UMAP visualization of dominant logic classifications.
"""

import pandas as pd
import webbrowser
import os

def demonstrate_interactive_features():
    """Demonstrate the key features of the interactive UMAP plots."""
    
    print("="*70)
    print("INTERACTIVE UMAP VISUALIZATION DEMO")
    print("="*70)
    
    # Load and summarize the data
    csv_path = "/Users/tamastorkoly/Documents/Doc-MB15/Projects/quantize_culture/umap_2d_data.csv"
    df = pd.read_csv(csv_path)
    
    print(f"\n📊 DATA OVERVIEW:")
    print(f"   • Total passages: {len(df)}")
    print(f"   • Classes: {', '.join(df['Dominant_Logic'].unique())}")
    print(f"   • Dimensions: 2D UMAP reduction from high-dimensional OpenAI embeddings")
    
    print(f"\n📈 CLASS DISTRIBUTION:")
    class_counts = df['Dominant_Logic'].value_counts()
    for class_name, count in class_counts.items():
        percentage = (count / len(df)) * 100
        print(f"   • {class_name}: {count} passages ({percentage:.1f}%)")
    
    print(f"\n🎯 INTERACTIVE FEATURES:")
    print(f"   • Hover over any point to see the FULL passage text (no truncation)")
    print(f"   • Points are color-coded by dominant logic class")
    print(f"   • Star markers show class centroids")
    print(f"   • Zoom and pan to explore clusters")
    print(f"   • Click legend items to hide/show classes")
    print(f"   • Legend positioned in upper right corner")
    print(f"   • Long text is properly wrapped for readability")
    
    print(f"\n🔍 KEY INSIGHTS TO EXPLORE:")
    
    # Calculate some interesting insights
    centroids = df.groupby('Dominant_Logic')[['UMAP_1', 'UMAP_2']].mean()
    
    print(f"   • CERTAINTY cluster (blue): Tends to be positioned at ({centroids.loc['CERTAINTY', 'UMAP_1']:.1f}, {centroids.loc['CERTAINTY', 'UMAP_2']:.1f})")
    print(f"   • ENTREPRENEUR cluster (orange): Centers around ({centroids.loc['ENTREPRENEUR', 'UMAP_1']:.1f}, {centroids.loc['ENTREPRENEUR', 'UMAP_2']:.1f})")
    print(f"   • FINANCIAL PERFORMANCE FIRST cluster (green): Located at ({centroids.loc['FINANCIAL PERFORMANCE FIRST', 'UMAP_1']:.1f}, {centroids.loc['FINANCIAL PERFORMANCE FIRST', 'UMAP_2']:.1f})")
    
    # Find some extreme examples
    print(f"\n📝 SAMPLE PASSAGES TO EXAMINE:")
    
    # Get one example from each class
    for class_name in df['Dominant_Logic'].unique():
        class_data = df[df['Dominant_Logic'] == class_name]
        sample = class_data.iloc[0]
        print(f"\n   {class_name}:")
        print(f"   Location: ({sample['UMAP_1']:.2f}, {sample['UMAP_2']:.2f})")
        print(f"   Preview: \"{sample['Passage'][:80]}...\"")
    
    print(f"\n🌐 HOW TO USE:")
    print(f"   1. Open one of these files in your web browser:")
    print(f"      • interactive_umap_plot.html (basic version)")
    print(f"      • enhanced_interactive_umap_plot.html (with centroids)")
    print(f"   2. Hover over points to read the COMPLETE passages (no truncation)")
    print(f"   3. Look for clustering patterns and outliers")
    print(f"   4. Use the legend (upper right) to focus on specific classes")
    print(f"   5. Zoom and pan to explore different regions in detail")
    
    print("="*70)
    
    return {
        'total_points': len(df),
        'classes': df['Dominant_Logic'].unique().tolist(),
        'centroids': centroids.to_dict('index'),
        'class_counts': class_counts.to_dict()
    }

def open_visualizations():
    """Open both interactive visualizations in the browser."""
    base_path = "/Users/tamastorkoly/Documents/Doc-MB15/Projects/quantize_culture"
    
    basic_path = os.path.join(base_path, "interactive_umap_plot.html")
    enhanced_path = os.path.join(base_path, "enhanced_interactive_umap_plot.html")
    
    print(f"\n🚀 OPENING INTERACTIVE VISUALIZATIONS...")
    
    if os.path.exists(enhanced_path):
        webbrowser.open(f"file://{enhanced_path}")
        print(f"   ✅ Opened enhanced interactive plot")
    else:
        print(f"   ❌ Enhanced plot not found at {enhanced_path}")
    
    if os.path.exists(basic_path):
        # Open basic version in a new tab after a small delay
        import time
        time.sleep(2)
        webbrowser.open(f"file://{basic_path}")
        print(f"   ✅ Opened basic interactive plot")
    else:
        print(f"   ❌ Basic plot not found at {basic_path}")

def main():
    """Main demo function."""
    
    # Show the demonstration
    demo_results = demonstrate_interactive_features()
    
    # Ask if user wants to open the visualizations
    print(f"\n🤔 Would you like to open the interactive visualizations in your browser?")
    print(f"   (The plots should already be visible in VS Code's Simple Browser)")
    
    # For now, just show the file paths
    print(f"\n📁 Interactive files are available at:")
    print(f"   • interactive_umap_plot.html")
    print(f"   • enhanced_interactive_umap_plot.html")
    
    print(f"\n✨ PIPELINE COMPLETE! ✨")
    print(f"You now have fully interactive UMAP visualizations where you can:")
    print(f"   • Hover over any point to see the passage text")
    print(f"   • Explore the relationship between embeddings and classifications")
    print(f"   • Identify clusters and outliers in the semantic space")
    
    return demo_results

if __name__ == "__main__":
    main()
