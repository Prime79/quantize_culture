from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os
import openai

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

if __name__ == "__main__":
    test_qdrant_connection()
    test_openai_api_key()
