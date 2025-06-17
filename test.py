from qdrant_client import QdrantClient

# Connect to local Qdrant instance
def test_qdrant_connection():
    client = QdrantClient(host="localhost", port=6333)
    try:
        # List collections as a simple test
        collections = client.get_collections()
        print(f"Qdrant is running. Collections: {collections}")
        return True
    except Exception as e:
        print(f"Failed to connect to Qdrant: {e}")
        return False

if __name__ == "__main__":
    test_qdrant_connection()
