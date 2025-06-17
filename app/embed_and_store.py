import os
from dotenv import load_dotenv
import openai
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, Distance, VectorParams, CollectionStatus

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")  # Use 'qdrant' as default for Docker network
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "embeddings")

openai.api_key = OPENAI_API_KEY

# Initialize Qdrant client
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# Ensure collection exists
def ensure_collection():
    collections = client.get_collections().collections
    if not any(c.name == COLLECTION_NAME for c in collections):
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )
        print(f"Created collection '{COLLECTION_NAME}' in Qdrant.")
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists.")

# Query OpenAI for embedding and write to Qdrant
def embed_and_store(sentence: str):
    # Get embedding from OpenAI
    response = openai.embeddings.create(
        input=sentence,
        model="text-embedding-ada-002"
    )
    embedding = response.data[0].embedding
    # Write to Qdrant
    point = PointStruct(
        id=None,  # Let Qdrant auto-assign
        vector=embedding,
        payload={"sentence": sentence}
    )
    client.upsert(collection_name=COLLECTION_NAME, points=[point])
    print(f"Stored embedding for: '{sentence}'")

if __name__ == "__main__":
    ensure_collection()
    test_sentence = "Our company values innovation and teamwork."
    embed_and_store(test_sentence)
