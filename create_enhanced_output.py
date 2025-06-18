#!/usr/bin/env python3
"""
Quick script to add cluster information to all sentences and then create out_01.json
"""

import json
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.extract import DataExtractorAnalyzer

def run_clustering_and_create_output():
    """Run clustering analysis and create output JSON with cluster information."""
    print("🚀 Running clustering analysis and creating out_01.json\n")
    
    # Step 1: Run clustering analysis on all data
    print("🔍 Step 1: Running clustering analysis...")
    analyzer = DataExtractorAnalyzer()
    
    # Extract all data
    print("   📊 Extracting all data...")
    data = analyzer.extract_data()  # Get all data
    print(f"   ✅ Extracted {len(data)} sentences")
    
    # Run dimensionality reduction and clustering
    print("   🎯 Reducing dimensions...")
    analyzer.reduce_dimensions(n_components=2, n_neighbors=15, min_dist=0.1)
    
    print("   🔍 Clustering...")
    analyzer.cluster_data(min_cluster_size=8, min_samples=5)
    
    # Store cluster results back to database
    print("   💾 Storing cluster results...")
    analyzer.store_clusters_to_database()
    
    print(f"   ✅ Clustering complete!")
    
    # Step 2: Load original JSON structure
    print("\n📁 Step 2: Loading original data structure...")
    with open('data_01.json', 'r') as f:
        original_data = json.load(f)
    print("   ✅ Original data loaded")
    
    # Step 3: Create mapping of sentences to clusters
    print("\n🔄 Step 3: Creating sentence-to-cluster mapping...")
    sentence_to_cluster = {}
    
    for _, row in analyzer.data.iterrows():
        sentence = row['sentence']
        cluster_id = row['cluster']
        cluster_name = f"cluster_{cluster_id}" if cluster_id >= 0 else "noise"
        
        sentence_to_cluster[sentence] = {
            'cluster_id': int(cluster_id),
            'cluster_name': cluster_name
        }
    
    print(f"   ✅ Mapped {len(sentence_to_cluster)} sentences to clusters")
    
    # Step 4: Create enhanced JSON structure
    print("\n📝 Step 4: Creating enhanced JSON structure...")
    output_data = {}
    total_sentences = 0
    sentences_with_clusters = 0
    
    for main_category, subcategories in original_data.items():
        output_data[main_category] = {}
        
        for subcategory, sentences in subcategories.items():
            output_data[main_category][subcategory] = []
            
            for sentence in sentences:
                total_sentences += 1
                
                # Find cluster information for this sentence
                cluster_info = sentence_to_cluster.get(sentence)
                
                if cluster_info:
                    # Create enhanced sentence object with cluster info
                    enhanced_sentence = {
                        "sentence": sentence,
                        "cluster_id": cluster_info['cluster_id'],
                        "cluster_name": cluster_info['cluster_name']
                    }
                    sentences_with_clusters += 1
                else:
                    # Sentence not found in clustering results
                    enhanced_sentence = {
                        "sentence": sentence,
                        "cluster_id": None,
                        "cluster_name": "not_processed"
                    }
                
                output_data[main_category][subcategory].append(enhanced_sentence)
    
    print(f"   ✅ Enhanced {sentences_with_clusters}/{total_sentences} sentences")
    
    # Step 5: Save enhanced data
    print("\n💾 Step 5: Saving enhanced data to out_01.json...")
    with open('out_01.json', 'w') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print("   ✅ File saved successfully")
    
    # Step 6: Generate summary
    print("\n📊 Step 6: Generating cluster summary...")
    cluster_summary = analyzer.get_cluster_summary()
    
    print(f"\n{'='*60}")
    print(f"📊 CLUSTERING ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Total sentences in original data: {total_sentences}")
    print(f"Sentences successfully clustered: {sentences_with_clusters}")
    print(f"Number of clusters found: {len(cluster_summary[cluster_summary['cluster_id'] >= 0])}")
    print(f"Noise points: {len(cluster_summary[cluster_summary['cluster_id'] == -1])}")
    
    print(f"\n📝 CLUSTER DETAILS:")
    print(f"{'-'*60}")
    
    for _, row in cluster_summary.iterrows():
        if row['cluster_id'] >= 0:  # Skip noise for detailed view
            print(f"\n🎯 {row['cluster_name'].upper()}: {row['size']} sentences")
            # Show sample sentences
            for i, sentence in enumerate(row['sample_sentences'][:3], 1):
                print(f"   {i}. \"{sentence[:80]}...\"")
    
    print(f"\n{'='*60}")
    print(f"✅ SUCCESS: out_01.json created with cluster information!")
    print(f"📁 The file maintains the original structure with added cluster data")
    print(f"🔍 Each sentence now includes cluster_id and cluster_name fields")
    print(f"{'='*60}")

if __name__ == "__main__":
    run_clustering_and_create_output()
