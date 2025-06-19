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

def ensure_reference_collection(collection_name: str, overwrite: bool = True):
    """
    Ensure reference collection exists with proper cleanup.
    
    Args:
        collection_name: Name of the reference collection
        overwrite: If True, delete existing collection and recreate (default: True)
    """
    print(f"🔧 Managing reference collection '{collection_name}'...")
    
    collections = client.get_collections().collections
    collection_exists = any(c.name == collection_name for c in collections)
    
    if collection_exists:
        if overwrite:
            print(f"   🗑️  Deleting existing collection '{collection_name}'...")
            client.delete_collection(collection_name)
            print(f"   ✅ Existing collection deleted")
        else:
            print(f"   ⚠️  Collection '{collection_name}' already exists, keeping existing data")
            return
    
    # Create fresh collection
    print(f"   🆕 Creating fresh collection '{collection_name}'...")
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
    )
    print(f"   ✅ Reference collection '{collection_name}' ready")

def embed_and_store_to_reference_collection(sentences, collection_name: str, overwrite: bool = True):
    """
    Embed and store sentences to a specific reference collection with cleanup.
    
    Args:
        sentences: List of sentences to embed and store
        collection_name: Name of the reference collection
        overwrite: If True, recreate collection (default: True)
    """
    if not sentences:
        print("❌ No sentences provided for embedding.")
        return 0
    
    print(f"📊 Processing {len(sentences)} sentences for reference collection '{collection_name}'...")
    
    # Ensure clean reference collection
    ensure_reference_collection(collection_name, overwrite=overwrite)
    
    # Get embeddings from OpenAI in batch
    print(f"   🤖 Getting embeddings from OpenAI...")
    response = openai.embeddings.create(
        input=sentences,
        model="text-embedding-ada-002"
    )
    embeddings = [item.embedding for item in response.data]
    
    # Create points with unique IDs
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={"sentence": sentence}
        )
        for sentence, embedding in zip(sentences, embeddings)
    ]
    
    # Store in reference collection
    print(f"   💾 Storing {len(points)} embeddings...")
    client.upsert(collection_name=collection_name, points=points)
    
    # Verify storage
    collection_info = client.get_collection(collection_name)
    actual_count = collection_info.points_count
    
    print(f"   ✅ Stored {actual_count} embeddings in reference collection '{collection_name}'")
    
    if actual_count != len(sentences):
        print(f"   ⚠️  Warning: Expected {len(sentences)} but stored {actual_count}")
    
    return actual_count

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
