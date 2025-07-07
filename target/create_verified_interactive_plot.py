#!/usr/bin/env python3
"""
Generate Fresh Clean Interactive UMAP Plot

This script creates a completely new, verified interactive plot to fix
the coordinate mismatch issue. Includes extensive validation.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

def create_verified_interactive_plot():
    """Create a fresh, verified interactive UMAP plot."""
    
    print("🔄 CREATING FRESH VERIFIED INTERACTIVE PLOT")
    print("="*60)
    
    # Load and verify data
    csv_path = "/Users/tamastorkoly/Documents/Doc-MB15/Projects/quantize_culture/umap_2d_data.csv"
    df = pd.read_csv(csv_path)
    
    print(f"📊 Loaded {len(df)} records")
    
    # Verify the specific problematic coordinates
    test_coords = [
        (0.982, 1.395, "ideally you would form a team"),
        (8.351, 5.412, "put people who move fast"),
        (0.090, 3.294, "getting to the real pain point")
    ]
    
    print(f"\n🔍 VERIFICATION OF KEY COORDINATES:")
    for x, y, expected_text in test_coords:
        # Find closest match
        distances = ((df['UMAP_1'] - x)**2 + (df['UMAP_2'] - y)**2)**0.5
        closest_idx = distances.idxmin()
        closest_row = df.iloc[closest_idx]
        
        print(f"📍 ({x:.3f}, {y:.3f}): Expected '{expected_text}'")
        print(f"   ✅ Found: '{closest_row['Passage'][:50]}...'")
        print(f"   📏 Distance: {distances.min():.6f}")
        
        if expected_text.lower() in closest_row['Passage'].lower():
            print(f"   ✅ VERIFIED CORRECT")
        else:
            print(f"   ❌ MISMATCH DETECTED!")
        print()
    
    # Create clean data for plotting
    plot_data = df.copy()
    
    # Add a unique ID to each row for verification
    plot_data['Point_ID'] = range(len(plot_data))
    
    # Add truncated passage for display (but keep full for hover)
    plot_data['Display_Preview'] = plot_data['Passage'].apply(
        lambda x: x[:60] + "..." if len(x) > 60 else x
    )
    
    # Define colors
    color_map = {
        'CERTAINTY': '#1f77b4',           # Blue
        'ENTREPRENEUR': '#ff7f0e',        # Orange  
        'FINANCIAL PERFORMANCE FIRST': '#2ca02c'  # Green
    }
    
    print(f"🎨 Creating Plotly figure...")
    
    # Create the plot
    fig = go.Figure()
    
    # Add points for each class separately for better control
    for class_name in plot_data['Dominant_Logic'].unique():
        class_data = plot_data[plot_data['Dominant_Logic'] == class_name]
        
        fig.add_trace(go.Scatter(
            x=class_data['UMAP_1'],
            y=class_data['UMAP_2'],
            mode='markers',
            name=f'{class_name} (n={len(class_data)})',
            marker=dict(
                color=color_map[class_name],
                size=8,
                opacity=0.7,
                line=dict(width=1, color='white')
            ),
            text=class_data['Point_ID'],  # Add point ID for debugging
            customdata=class_data[['Point_ID', 'Dominant_Logic', 'Passage', 'Display_Preview']].values,
            hovertemplate='<b>Point ID: %{customdata[0]}</b><br>' +
                         '<b>Class: %{customdata[1]}</b><br>' +
                         'UMAP_1: %{x:.6f}<br>' +
                         'UMAP_2: %{y:.6f}<br>' +
                         '<br><b>Full Passage:</b><br>' +
                         '%{customdata[2]}<br>' +
                         '<extra></extra>',
            hoverlabel=dict(
                bgcolor="white",
                bordercolor="black",
                font_size=11,
                font_family="Arial",
                align="left"
            )
        ))
    
    # Update layout
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    fig.update_layout(
        title=dict(
            text=f'VERIFIED Interactive 2D UMAP Visualization<br><sub>Generated: {timestamp} | Point IDs included for verification</sub>',
            font=dict(size=16),
            x=0.5,
            xanchor='center'
        ),
        xaxis_title='UMAP Dimension 1',
        yaxis_title='UMAP Dimension 2',
        width=1200,
        height=800,
        font=dict(size=12),
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1
        )
    )
    
    # Update axes
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        showline=True,
        linewidth=1,
        linecolor='black',
        mirror=True
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        showline=True,
        linewidth=1,
        linecolor='black',
        mirror=True
    )
    
    # Save the verified plot
    output_path = f"/Users/tamastorkoly/Documents/Doc-MB15/Projects/quantize_culture/VERIFIED_interactive_umap_plot_{timestamp.replace(':', '-').replace(' ', '_')}.html"
    fig.write_html(output_path)
    
    print(f"💾 Saved verified plot to: {output_path}")
    
    # Also create a simple backup with standard name
    backup_path = "/Users/tamastorkoly/Documents/Doc-MB15/Projects/quantize_culture/VERIFIED_interactive_umap_plot.html"
    fig.write_html(backup_path)
    print(f"💾 Backup saved to: {backup_path}")
    
    # Export verification data
    verification_df = plot_data[['Point_ID', 'UMAP_1', 'UMAP_2', 'Dominant_Logic', 'Passage']].copy()
    verification_df.to_csv('verification_data_with_point_ids.csv', index=False)
    print(f"💾 Verification data exported to: verification_data_with_point_ids.csv")
    
    print(f"\n✅ VERIFICATION COMPLETE!")
    print(f"📍 Key coordinates to test in the new plot:")
    print(f"   • Point at (0.982, 1.395) should show 'ideally you would form a team...'")
    print(f"   • Point at (8.351, 5.412) should show 'put people who move fast...'")
    print(f"   • Point at (0.090, 3.294) should show 'getting to the real pain point...'")
    
    return fig, output_path

def main():
    """Generate the verified plot."""
    fig, output_path = create_verified_interactive_plot()
    
    print(f"\n🎯 NEXT STEPS:")
    print(f"1. Close all existing browser tabs with UMAP plots")
    print(f"2. Clear your browser cache")
    print(f"3. Open the new verified plot: {output_path}")
    print(f"4. Test the coordinates mentioned above")
    print(f"5. Each point now shows a Point ID for verification")

if __name__ == "__main__":
    main()
