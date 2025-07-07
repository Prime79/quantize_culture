#!/usr/bin/env python3
"""
Visualize the class distribution of dominant logic categories.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from qdrant_client import QdrantClient
import numpy as np

def main():
    print("📊 Analyzing Dominant Logic Class Distribution")
    print("=" * 50)
    
    # Connect to Qdrant
    client = QdrantClient(host="localhost", port=6333)
    collection_name = "target_test_umap10d"
    
    # Extract data
    points, _ = client.scroll(
        collection_name=collection_name,
        limit=10000,
        with_payload=True,
        with_vectors=True
    )
    
    # Filter valid dominant logic labels
    valid_labels = []
    for point in points:
        dominant_logic = point.payload.get('dominant_logic', '').strip()
        if dominant_logic and dominant_logic.lower() not in ['', 'none', 'null', 'unknown']:
            valid_labels.append(dominant_logic)
    
    # Create distribution
    df = pd.DataFrame({'dominant_logic': valid_labels})
    class_counts = df['dominant_logic'].value_counts()
    
    print(f"📈 Total valid samples: {len(valid_labels)}")
    print(f"🏷️  Unique classes: {len(class_counts)}")
    print(f"\n📊 Complete Class Distribution:")
    print("=" * 60)
    
    # Print detailed distribution
    for i, (logic, count) in enumerate(class_counts.items()):
        percentage = count / len(valid_labels) * 100
        print(f"{i+1:2d}. {logic:35} | {count:3d} samples ({percentage:5.1f}%)")
    
    # Group by frequency tiers
    print(f"\n🎯 Frequency Tiers:")
    print("=" * 40)
    
    high_freq = class_counts[class_counts >= 20]
    medium_freq = class_counts[(class_counts >= 5) & (class_counts < 20)]
    low_freq = class_counts[(class_counts >= 2) & (class_counts < 5)]
    single_sample = class_counts[class_counts == 1]
    
    print(f"High Frequency (≥20): {len(high_freq)} classes, {high_freq.sum()} samples ({high_freq.sum()/len(valid_labels)*100:.1f}%)")
    for logic, count in high_freq.items():
        print(f"  • {logic}: {count}")
    
    print(f"\nMedium Frequency (5-19): {len(medium_freq)} classes, {medium_freq.sum()} samples ({medium_freq.sum()/len(valid_labels)*100:.1f}%)")
    for logic, count in medium_freq.items():
        print(f"  • {logic}: {count}")
    
    print(f"\nLow Frequency (2-4): {len(low_freq)} classes, {low_freq.sum()} samples ({low_freq.sum()/len(valid_labels)*100:.1f}%)")
    for logic, count in low_freq.items():
        print(f"  • {logic}: {count}")
    
    print(f"\nSingle Sample (1): {len(single_sample)} classes, {single_sample.sum()} samples ({single_sample.sum()/len(valid_labels)*100:.1f}%)")
    for logic, count in single_sample.items():
        print(f"  • {logic}: {count}")
    
    # Create visualization
    plt.figure(figsize=(14, 8))
    
    # Top 15 classes for readability
    top_classes = class_counts.head(15)
    
    # Create bar plot
    bars = plt.bar(range(len(top_classes)), top_classes.values, color='steelblue', alpha=0.7)
    
    # Add value labels on bars
    for i, (logic, count) in enumerate(top_classes.items()):
        plt.text(i, count + 0.5, str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.title('Dominant Logic Class Distribution (Top 15 Classes)', fontsize=16, fontweight='bold')
    plt.xlabel('Dominant Logic Categories', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.xticks(range(len(top_classes)), top_classes.index, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
    print(f"\n💾 Distribution plot saved as 'class_distribution.png'")
    plt.show()
    
    # Calculate Gini coefficient for imbalance measure
    sorted_counts = sorted(class_counts.values, reverse=True)
    n = len(sorted_counts)
    cumsum = np.cumsum(sorted_counts)
    gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
    
    print(f"\n📊 Class Imbalance Metrics:")
    print(f"   • Gini Coefficient: {gini:.3f} (0=perfect balance, 1=maximum imbalance)")
    print(f"   • Most common class ratio: {class_counts.iloc[0]/len(valid_labels):.1%}")
    print(f"   • Classes with <5 samples: {len(class_counts[class_counts < 5])}/{len(class_counts)} ({len(class_counts[class_counts < 5])/len(class_counts)*100:.1f}%)")

if __name__ == "__main__":
    main()
