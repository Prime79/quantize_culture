import os
from dotenv import load_dotenv
import openai
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, Distance, VectorParams, CollectionStatus
import uuid

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

def embed_and_store_bulk(sentences, qdrant_client=None, collection_name=None):
    """
    Embeds and stores a list of sentences in Qdrant in bulk.
    Optionally accepts a QdrantClient and collection name for flexibility.
    """
    if not sentences:
        print("No sentences provided for bulk embedding.")
        return
    if qdrant_client is None:
        qdrant_client = client
    if collection_name is None:
        collection_name = COLLECTION_NAME
    # Get embeddings from OpenAI in batch
    response = openai.embeddings.create(
        input=sentences,
        model="text-embedding-ada-002"
    )
    embeddings = [item.embedding for item in response.data]
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={"sentence": sentence}
        )
        for sentence, embedding in zip(sentences, embeddings)
    ]
    qdrant_client.upsert(collection_name=collection_name, points=points)
    print(f"Stored embeddings for {len(sentences)} sentences in collection '{collection_name}'.")

if __name__ == "__main__":
    ensure_collection()
    test_sentence = "Our company values innovation and teamwork."
    embed_and_store(test_sentence)
    # Example bulk usage
    test_sentences = [
        "We encourage open communication.",
        "Customer satisfaction is our top priority.",
        "We value diversity and inclusion."
    ]
    embed_and_store_bulk(test_sentences)
