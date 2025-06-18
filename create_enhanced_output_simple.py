#!/usr/bin/env python3
"""
Create enhanced output JSON with cluster information.
Takes the original data_01.json and adds cluster names to each sentence.
"""

import json
from qdrant_client import QdrantClient

def create_enhanced_output():
    """Create enhanced output JSON with cluster information."""
    print("📊 Creating enhanced output JSON with cluster information...")
    
    # Load original JSON
    print("📁 Loading original data_01.json...")
    with open('data_01.json', 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    print(f"   ✅ Loaded {len(original_data)} categories")
    
    # Connect to Qdrant and get all points with cluster info
    print("🔍 Connecting to Qdrant database...")
    client = QdrantClient(host="localhost", port=6333)
    
    # Get all points from database in batches
    print("📊 Extracting cluster information from database...")
    sentence_to_cluster = {}
    offset = None
    batch_count = 0
    
    while True:
        try:
            result = client.scroll(
                collection_name="company_culture_embeddings",
                limit=50,  # Smaller batches
                offset=offset,
                with_payload=True,
                with_vectors=False  # Don't need vectors, just payload
            )
            
            points, next_offset = result
            if not points:
                break
            
            batch_count += 1
            print(f"   Processing batch {batch_count} ({len(points)} points)...")
            
            for point in points:
                sentence = point.payload.get('sentence', '')
                cluster_id = point.payload.get('cluster', -1)
                cluster_name = point.payload.get('cluster_name', f"cluster_{cluster_id}" if cluster_id >= 0 else "noise")
                
                sentence_to_cluster[sentence] = {
                    'cluster_id': cluster_id,
                    'cluster_name': cluster_name
                }
            
            offset = next_offset
            if not next_offset:
                break
                
        except Exception as e:
            print(f"   ⚠️  Error in batch {batch_count}: {e}")
            break
    
    print(f"   ✅ Extracted cluster info for {len(sentence_to_cluster)} sentences")
    
    # Create enhanced output
    print("🔧 Creating enhanced JSON structure...")
    enhanced_data = {}
    total_sentences = 0
    clustered_sentences = 0
    
    for category, subcategories in original_data.items():
        enhanced_category = {}
        
        for subcategory, sentences in subcategories.items():
            enhanced_sentences = []
            
            for sentence in sentences:
                enhanced_sentence = {
                    'sentence': sentence
                }
                
                # Add cluster information if available
                if sentence in sentence_to_cluster:
                    cluster_info = sentence_to_cluster[sentence]
                    enhanced_sentence['cluster_id'] = cluster_info['cluster_id']
                    enhanced_sentence['cluster_name'] = cluster_info['cluster_name']
                    clustered_sentences += 1
                else:
                    # No cluster info available
                    enhanced_sentence['cluster_id'] = -1
                    enhanced_sentence['cluster_name'] = "not_clustered"
                
                enhanced_sentences.append(enhanced_sentence)
                total_sentences += 1
            
            enhanced_category[subcategory] = enhanced_sentences
        
        enhanced_data[category] = enhanced_category
    
    # Save enhanced output
    print("💾 Saving enhanced output to out_01.json...")
    with open('out_01.json', 'w', encoding='utf-8') as f:
        json.dump(enhanced_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Enhanced output created successfully!")
    print(f"   📊 Total sentences: {total_sentences}")
    print(f"   🎯 Sentences with clusters: {clustered_sentences}")
    print(f"   📁 Saved to: out_01.json")
    
    # Show cluster distribution
    cluster_counts = {}
    for sentence, info in sentence_to_cluster.items():
        cluster_name = info['cluster_name']
        cluster_counts[cluster_name] = cluster_counts.get(cluster_name, 0) + 1
    
    print(f"\n📈 Cluster Distribution:")
    for cluster_name, count in sorted(cluster_counts.items()):
        print(f"   • {cluster_name}: {count} sentences")
    
    return enhanced_data

if __name__ == "__main__":
    create_enhanced_output()
