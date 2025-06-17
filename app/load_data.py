import json
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from .embed_and_store import embed_and_store_bulk

# Load environment variables
load_dotenv()

def load_sentences_from_json(file_path):
    """
    Load sentences from the JSON file and extract all sentences from inner lists.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    sentences = []
    for category, subcategories in data.items():
        for subcategory, sentence_list in subcategories.items():
            sentences.extend(sentence_list)
    
    print(f"Extracted {len(sentences)} sentences from {file_path}")
    return sentences

def load_data_to_qdrant():
    """
    Load sentences from data_01.json and store them in Qdrant.
    """
    # Load sentences from JSON
    sentences = load_sentences_from_json('data_01.json')
    
    # Create local Qdrant client
    client = QdrantClient(host="localhost", port=6333)
    collection_name = "company_culture_embeddings"
    
    # Ensure collection exists
    collections = client.get_collections().collections
    if not any(c.name == collection_name for c in collections):
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )
        print(f"Created collection '{collection_name}'")
    
    # Embed and store in bulk
    embed_and_store_bulk(sentences, qdrant_client=client, collection_name=collection_name)
    
    return len(sentences), collection_name

def test_data_loaded(collection_name, expected_count):
    """
    Test if the data was loaded correctly into Qdrant.
    """
    client = QdrantClient(host="localhost", port=6333)
    
    # Get collection info
    collection_info = client.get_collection(collection_name)
    print(f"Collection '{collection_name}' has {collection_info.points_count} points")
    
    # Verify count matches expected
    if collection_info.points_count == expected_count:
        print("✅ Data loaded successfully!")
        return True
    else:
        print(f"❌ Expected {expected_count} points, but found {collection_info.points_count}")
        return False

if __name__ == "__main__":
    sentence_count, collection_name = load_data_to_qdrant()
    test_data_loaded(collection_name, sentence_count)
