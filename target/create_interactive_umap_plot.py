#!/usr/bin/env python3
"""
Interactive UMAP Visualization with Plotly

Creates an interactive 2D UMAP scatterplot where hovering over points 
shows the corresponding passage/sentence text. This enables exploration 
of the embeddings and their relationship to dominant logic classifications.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot
import numpy as np

def load_umap_data(csv_path):
    """Load UMAP data from CSV file."""
    print(f"Loading UMAP data from {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} data points")
    print(f"Classes: {df['Dominant_Logic'].unique()}")
    print(f"Class distribution:")
    print(df['Dominant_Logic'].value_counts())
    return df

def wrap_text_for_hover(text, max_line_length=80):
    """Wrap long text for better display in hover tooltips."""
    import textwrap
    # Split into lines and wrap each line
    lines = text.split('\n')
    wrapped_lines = []
    for line in lines:
        if len(line) <= max_line_length:
            wrapped_lines.append(line)
        else:
            wrapped_lines.extend(textwrap.wrap(line, width=max_line_length))
    return '<br>'.join(wrapped_lines)

def create_interactive_umap_plot(df, output_path="interactive_umap_plot.html"):
    """Create an interactive UMAP scatterplot with Plotly."""
    
    # Create wrapped text for better hover display
    df['Passage_Wrapped'] = df['Passage'].apply(lambda x: wrap_text_for_hover(x, 80))
    
    # Define colors for each class
    color_map = {
        'CERTAINTY': '#1f77b4',           # Blue
        'ENTREPRENEUR': '#ff7f0e',        # Orange  
        'FINANCIAL PERFORMANCE FIRST': '#2ca02c'  # Green
    }
    
    # Create the interactive scatter plot
    fig = px.scatter(
        df, 
        x='UMAP_1', 
        y='UMAP_2',
        color='Dominant_Logic',
        color_discrete_map=color_map,
        hover_data={
            'UMAP_1': ':.3f',
            'UMAP_2': ':.3f',
            'Dominant_Logic': True,
            'Passage': False,  # Don't show in hover by default
            'Passage_Wrapped': False  # Don't show in hover by default
        },
        title='Interactive 2D UMAP Visualization of Dominant Logic Classifications<br><sub>Hover over points to see passages</sub>',
        labels={
            'UMAP_1': 'UMAP Dimension 1',
            'UMAP_2': 'UMAP Dimension 2',
            'Dominant_Logic': 'Dominant Logic Class'
        },
        width=1000,
        height=700
    )
    
    # Customize hover template to show full passage text (no truncation)
    fig.update_traces(
        hovertemplate='<b>%{customdata[2]}</b><br>' +
                      'UMAP_1: %{x:.3f}<br>' +
                      'UMAP_2: %{y:.3f}<br>' +
                      '<br><b>Passage:</b><br>' +
                      '%{customdata[3]}<br>' +
                      '<extra></extra>',
        customdata=df[['UMAP_1', 'UMAP_2', 'Dominant_Logic', 'Passage_Wrapped']].values,
        hoverlabel=dict(
            bgcolor="white",
            bordercolor="black",
            font_size=12,
            font_family="Arial",
            align="left",
            namelength=-1  # Show full text without truncation
        )
    )
    
    # Update layout for better aesthetics
    fig.update_layout(
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
        ),
        title=dict(
            font=dict(size=16),
            x=0.5,
            xanchor='center'
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
    
    # Save as HTML
    print(f"Saving interactive plot to {output_path}")
    fig.write_html(output_path)
    print(f"Interactive plot saved! Open {output_path} in your browser to explore.")
    
    # Also show in default browser if possible
    try:
        fig.show()
        print("Plot opened in default browser.")
    except Exception as e:
        print(f"Could not open in browser automatically: {e}")
        print(f"Please manually open {output_path} in your browser.")
    
    return fig

def create_enhanced_interactive_plot(df, output_path="enhanced_interactive_umap_plot.html"):
    """Create an enhanced version with additional statistics and annotations."""
    
    # Calculate class centroids
    centroids = df.groupby('Dominant_Logic')[['UMAP_1', 'UMAP_2']].mean()
    
    # Create wrapped text for better hover display
    df['Passage_Wrapped'] = df['Passage'].apply(lambda x: wrap_text_for_hover(x, 80))
    
    # Define colors for each class
    color_map = {
        'CERTAINTY': '#1f77b4',           # Blue
        'ENTREPRENEUR': '#ff7f0e',        # Orange  
        'FINANCIAL PERFORMANCE FIRST': '#2ca02c'  # Green
    }
    
    # Create figure with subplots
    fig = go.Figure()
    
    # Add scatter points for each class
    for class_name in df['Dominant_Logic'].unique():
        class_data = df[df['Dominant_Logic'] == class_name]
        
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
            customdata=class_data[['Dominant_Logic', 'Passage_Wrapped']].values,
            hovertemplate='<b>%{customdata[0]}</b><br>' +
                         'UMAP_1: %{x:.3f}<br>' +
                         'UMAP_2: %{y:.3f}<br>' +
                         '<br><b>Passage:</b><br>' +
                         '%{customdata[1]}<br>' +
                         '<extra></extra>',
            hoverlabel=dict(
                bgcolor="white",
                bordercolor="black",
                font_size=12,
                font_family="Arial",
                align="left",
                namelength=-1  # Show full text without truncation
            )
        ))
    
    # Add centroid markers
    for class_name, centroid in centroids.iterrows():
        fig.add_trace(go.Scatter(
            x=[centroid['UMAP_1']],
            y=[centroid['UMAP_2']],
            mode='markers',
            name=f'{class_name} Centroid',
            marker=dict(
                color=color_map[class_name],
                size=15,
                symbol='star',
                line=dict(width=2, color='black')
            ),
            showlegend=False,
            hovertemplate=f'<b>{class_name} Centroid</b><br>' +
                         'UMAP_1: %{x:.3f}<br>' +
                         'UMAP_2: %{y:.3f}<br>' +
                         '<extra></extra>'
        ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text='Enhanced Interactive 2D UMAP Visualization<br><sub>Hover over points to see passages • Stars show class centroids</sub>',
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
    
    # Save as HTML
    print(f"Saving enhanced interactive plot to {output_path}")
    fig.write_html(output_path)
    print(f"Enhanced interactive plot saved! Open {output_path} in your browser to explore.")
    
    return fig

def print_summary_stats(df):
    """Print summary statistics about the data."""
    print("\n" + "="*60)
    print("UMAP DATA SUMMARY")
    print("="*60)
    
    print(f"Total data points: {len(df)}")
    print(f"UMAP dimensions: 2D (UMAP_1, UMAP_2)")
    
    print(f"\nClass distribution:")
    class_counts = df['Dominant_Logic'].value_counts()
    for class_name, count in class_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {class_name}: {count} ({percentage:.1f}%)")
    
    print(f"\nUMAP coordinate ranges:")
    print(f"  UMAP_1: {df['UMAP_1'].min():.3f} to {df['UMAP_1'].max():.3f}")
    print(f"  UMAP_2: {df['UMAP_2'].min():.3f} to {df['UMAP_2'].max():.3f}")
    
    print(f"\nClass centroids:")
    centroids = df.groupby('Dominant_Logic')[['UMAP_1', 'UMAP_2']].mean()
    for class_name, centroid in centroids.iterrows():
        print(f"  {class_name}: ({centroid['UMAP_1']:.3f}, {centroid['UMAP_2']:.3f})")
    
    print("="*60)

def main():
    """Main function to create interactive UMAP visualizations."""
    
    # Load data
    csv_path = "/Users/tamastorkoly/Documents/Doc-MB15/Projects/quantize_culture/umap_2d_data.csv"
    df = load_umap_data(csv_path)
    
    # Print summary statistics
    print_summary_stats(df)
    
    # Create basic interactive plot
    print("\nCreating basic interactive UMAP plot...")
    basic_fig = create_interactive_umap_plot(
        df, 
        output_path="/Users/tamastorkoly/Documents/Doc-MB15/Projects/quantize_culture/interactive_umap_plot.html"
    )
    
    # Create enhanced interactive plot
    print("\nCreating enhanced interactive UMAP plot...")
    enhanced_fig = create_enhanced_interactive_plot(
        df, 
        output_path="/Users/tamastorkoly/Documents/Doc-MB15/Projects/quantize_culture/enhanced_interactive_umap_plot.html"
    )
    
    print("\n" + "="*60)
    print("INTERACTIVE VISUALIZATION COMPLETE!")
    print("="*60)
    print("Two interactive HTML files have been created:")
    print("1. interactive_umap_plot.html - Basic interactive plot")
    print("2. enhanced_interactive_umap_plot.html - Enhanced with centroids")
    print("\nOpen either file in your web browser to explore the data.")
    print("Hover over any point to see the corresponding passage text.")
    print("="*60)

if __name__ == "__main__":
    main()
