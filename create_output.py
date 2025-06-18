#!/usr/bin/env python3
"""
Script to create out_01.json with cluster information added to the original data structure.
"""

import json
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.extract import DataExtractorAnalyzer

def extract_sentences_with_clusters():
    """Extract sentences with their cluster information using DataExtractorAnalyzer."""
    print("📊 Extracting sentences with cluster information from database...")
    
    analyzer = DataExtractorAnalyzer()
    
    # Extract all data with cluster information
    data = analyzer.extract_data(limit=200)  # Use reasonable limit
    
    print(f"   ✅ Retrieved {len(data)} sentences")
    
    # Create a mapping of sentence -> cluster info
    sentence_to_cluster = {}
    for _, row in data.iterrows():
        sentence = row['sentence']
        # Check if we have cluster information in the database
        if 'cluster' in row and row['cluster'] is not None:
            cluster_id = row['cluster']
            cluster_name = f"cluster_{cluster_id}" if cluster_id >= 0 else "noise"
        else:
            cluster_id = None
            cluster_name = "unprocessed"
        
        sentence_to_cluster[sentence] = {
            'cluster_id': cluster_id,
            'cluster_name': cluster_name
        }
    
    return sentence_to_cluster

def load_original_data():
    """Load the original JSON data."""
    print("📁 Loading original data from data_01.json...")
    
    with open('data_01.json', 'r') as f:
        data = json.load(f)
    
    print("   ✅ Original data loaded successfully")
    return data

def create_output_with_clusters(original_data, sentence_to_cluster):
    """Create the output JSON with cluster information added."""
    print("🔄 Adding cluster information to original data structure...")
    
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
                
                if cluster_info and cluster_info['cluster_id'] is not None:
                    # Create enhanced sentence object with cluster info
                    enhanced_sentence = {
                        "sentence": sentence,
                        "cluster_id": cluster_info['cluster_id'],
                        "cluster_name": cluster_info['cluster_name']
                    }
                    sentences_with_clusters += 1
                else:
                    # Sentence without cluster info (maybe not processed yet)
                    enhanced_sentence = {
                        "sentence": sentence,
                        "cluster_id": None,
                        "cluster_name": "unprocessed"
                    }
                
                output_data[main_category][subcategory].append(enhanced_sentence)
    
    print(f"   ✅ Enhanced {sentences_with_clusters}/{total_sentences} sentences with cluster information")
    return output_data

def save_output_file(output_data, filename='out_01.json'):
    """Save the enhanced data to output file."""
    print(f"💾 Saving enhanced data to {filename}...")
    
    with open(filename, 'w') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"   ✅ File saved successfully as {filename}")

def generate_cluster_summary(output_data):
    """Generate a summary of clusters found in the data."""
    print("\n📊 Cluster Summary:")
    print("=" * 50)
    
    cluster_counts = {}
    total_sentences = 0
    
    for main_category, subcategories in output_data.items():
        for subcategory, sentences in subcategories.items():
            for sentence_obj in sentences:
                total_sentences += 1
                cluster_name = sentence_obj.get('cluster_name', 'unknown')
                cluster_counts[cluster_name] = cluster_counts.get(cluster_name, 0) + 1
    
    print(f"Total sentences: {total_sentences}")
    print(f"Cluster distribution:")
    
    for cluster_name, count in sorted(cluster_counts.items()):
        percentage = (count / total_sentences) * 100
        print(f"   • {cluster_name}: {count} sentences ({percentage:.1f}%)")
    
    # Show sample sentences from each cluster
    print(f"\n📝 Sample sentences by cluster:")
    print("=" * 50)
    
    cluster_samples = {}
    for main_category, subcategories in output_data.items():
        for subcategory, sentences in subcategories.items():
            for sentence_obj in sentences:
                cluster_name = sentence_obj.get('cluster_name', 'unknown')
                if cluster_name not in cluster_samples:
                    cluster_samples[cluster_name] = []
                if len(cluster_samples[cluster_name]) < 3:  # Show max 3 samples per cluster
                    cluster_samples[cluster_name].append(sentence_obj['sentence'])
    
    for cluster_name, samples in sorted(cluster_samples.items()):
        print(f"\n🎯 {cluster_name.upper()}:")
        for i, sample in enumerate(samples, 1):
            print(f"   {i}. \"{sample[:80]}...\"")

def main():
    """Main function to create the enhanced JSON file."""
    print("🚀 Creating out_01.json with cluster information\n")
    
    try:
        # Extract cluster information from database
        sentence_to_cluster = extract_sentences_with_clusters()
        
        # Load original data
        original_data = load_original_data()
        
        # Create enhanced data with clusters
        output_data = create_output_with_clusters(original_data, sentence_to_cluster)
        
        # Save to output file
        save_output_file(output_data)
        
        # Generate summary
        generate_cluster_summary(output_data)
        
        print(f"\n🎉 Successfully created out_01.json!")
        print(f"📁 The file contains the same structure as data_01.json")
        print(f"✨ Each sentence now includes cluster_id and cluster_name")
        print(f"🔍 Use this file to analyze cultural themes by cluster")
        
    except Exception as e:
        print(f"❌ Error creating output file: {str(e)}")
        raise

if __name__ == "__main__":
    main()
