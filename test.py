from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os
import openai
from qdrant_client.http.models import PointStruct, Distance, VectorParams, PointsSelector, PointIdsList
import uuid
from app.embed_and_store import QDRANT_HOST, QDRANT_PORT, ensure_collection, embed_and_store_bulk, client as app_client
from app.load_data import load_data_to_qdrant, test_data_loaded

# Connect to local Qdrant instance
def test_qdrant_connection():
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    print(f"Loaded OpenAI API key: {'SET' if openai_api_key else 'NOT SET'}")
    client = QdrantClient(host="localhost", port=6333)
    try:
        # List collections as a simple test
        collections = client.get_collections()
        print(f"Qdrant is running. Collections: {collections}")
        return True
    except Exception as e:
        print(f"Failed to connect to Qdrant: {e}")
        return False

def test_openai_api_key():
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("OpenAI API key is NOT SET.")
        return False
    openai.api_key = openai_api_key
    try:
        # Make a simple API call to test the key
        models = openai.models.list()
        print(f"OpenAI API key is valid. Available models: {[m.id for m in models.data][:3]} ...")
        return True
    except Exception as e:
        print(f"OpenAI API key test failed: {e}")
        return False

def test_embed_and_store_and_remove():
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai.api_key = openai_api_key
    client = QdrantClient(host="localhost", port=6333)
    collection_name = "embeddings_test"
    # Ensure collection exists
    collections = client.get_collections().collections
    if not any(c.name == collection_name for c in collections):
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )
    # Embed and store
    sentence = "This is a test sentence for embedding."
    response = openai.embeddings.create(
        input=sentence,
        model="text-embedding-ada-002"
    )
    embedding = response.data[0].embedding
    point_id = str(uuid.uuid4())  # Use a valid UUID
    point = PointStruct(
        id=point_id,
        vector=embedding,
        payload={"sentence": sentence}
    )
    client.upsert(collection_name=collection_name, points=[point])
    print(f"Stored embedding for: '{sentence}' with id {point_id}")
    # Verify it exists
    search_result = client.scroll(collection_name=collection_name, scroll_filter={"must": [{"key": "sentence", "match": {"value": sentence}}]})
    assert any(p.payload.get("sentence") == sentence for p in search_result[0]), "Embedding not found in Qdrant!"
    print("Embedding found in Qdrant.")
    # Remove the test point
    client.delete(collection_name=collection_name, points_selector=PointIdsList(points=[point_id]))
    print(f"Removed test embedding with id {point_id}")

def test_embed_and_store_bulk_and_remove():
    load_dotenv()
    # Use local Qdrant client for test
    client = QdrantClient(host="localhost", port=6333)
    collection_name = "embeddings_test_bulk"
    # Ensure collection exists for test
    collections = client.get_collections().collections
    if not any(c.name == collection_name for c in collections):
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )
    # Test sentences
    sentences = [
        "Bulk test sentence one.",
        "Bulk test sentence two.",
        "Bulk test sentence three."
    ]
    # Embed and store in bulk
    embed_and_store_bulk(sentences, qdrant_client=client, collection_name=collection_name)
    # Verify all exist
    found_ids = []
    for sentence in sentences:
        search_result = client.scroll(collection_name=collection_name, scroll_filter={"must": [{"key": "sentence", "match": {"value": sentence}}]})
        assert any(p.payload.get("sentence") == sentence for p in search_result[0]), f"Sentence '{sentence}' not found in Qdrant!"
        found_ids.extend([p.id for p in search_result[0] if p.payload.get("sentence") == sentence])
    print("All bulk test sentences found in Qdrant.")
    # Remove all test points
    if found_ids:
        client.delete(collection_name=collection_name, points_selector=PointIdsList(points=found_ids))
        print(f"Removed bulk test embeddings with ids {found_ids}")
    else:
        print("No test points found to remove.")

def test_data_loading():
    """Test loading data from data_01.json into Qdrant"""
    sentence_count, collection_name = load_data_to_qdrant()
    test_data_loaded(collection_name, sentence_count)

if __name__ == "__main__":
    test_qdrant_connection()
    test_openai_api_key()
    test_embed_and_store_and_remove()
    test_embed_and_store_bulk_and_remove()
    test_data_loading()
