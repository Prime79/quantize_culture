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

# Contextualization settings
DOMAIN_CONTEXT_PREFIX = "Domain Logic example phrase: "
ENABLE_CONTEXTUALIZATION = True  # Set to False to disable contextualization globally

def contextualize_sentence(sentence: str, enable_context: bool = None) -> str:
    """
    Add domain context prefix to a sentence for better embedding quality.
    
    Args:
        sentence: Original sentence
        enable_context: Override global contextualization setting (optional)
    
    Returns:
        Contextualized sentence with domain prefix
    """
    if enable_context is None:
        enable_context = ENABLE_CONTEXTUALIZATION
    
    if enable_context:
        # Avoid double-prefixing if already contextualized
        if sentence.startswith(DOMAIN_CONTEXT_PREFIX):
            return sentence
        return f"{DOMAIN_CONTEXT_PREFIX}{sentence}"
    return sentence

def contextualize_sentences(sentences: list, enable_context: bool = None) -> list:
    """
    Add domain context prefix to a list of sentences.
    
    Args:
        sentences: List of original sentences
        enable_context: Override global contextualization setting (optional)
    
    Returns:
        List of contextualized sentences
    """
    return [contextualize_sentence(sentence, enable_context) for sentence in sentences]

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
def embed_and_store(sentence: str, enable_context: bool = None):
    # Contextualize sentence
    contextualized_sentence = contextualize_sentence(sentence, enable_context)
    
    # Get embedding from OpenAI
    response = openai.embeddings.create(
        input=contextualized_sentence,
        model="text-embedding-ada-002"
    )
    embedding = response.data[0].embedding
    # Write to Qdrant (store original sentence in payload, but embed contextualized version)
    point = PointStruct(
        id=str(uuid.uuid4()),  # Generate unique ID
        vector=embedding,
        payload={
            "sentence": sentence,  # Store original sentence
            "contextualized_sentence": contextualized_sentence  # Store contextualized version
        }
    )
    client.upsert(collection_name=COLLECTION_NAME, points=[point])
    print(f"Stored embedding for: '{sentence}' (contextualized)")

def embed_and_store_bulk(sentences, qdrant_client=None, collection_name=None, enable_context: bool = None):
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
    
    # Contextualize sentences
    contextualized_sentences = contextualize_sentences(sentences, enable_context)
    
    # Get embeddings from OpenAI in batch (use contextualized versions)
    response = openai.embeddings.create(
        input=contextualized_sentences,
        model="text-embedding-ada-002"
    )
    embeddings = [item.embedding for item in response.data]
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                "sentence": sentence,  # Store original sentence
                "contextualized_sentence": contextualized_sentence  # Store contextualized version
            }
        )
        for sentence, contextualized_sentence, embedding in zip(sentences, contextualized_sentences, embeddings)
    ]
    qdrant_client.upsert(collection_name=collection_name, points=points)
    print(f"Stored embeddings for {len(sentences)} sentences in collection '{collection_name}' (contextualized).")

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

def embed_and_store_to_reference_collection(sentences, collection_name: str, overwrite: bool = True, enable_context: bool = None):
    """
    Embed and store sentences to a specific reference collection with cleanup.
    
    Args:
        sentences: List of sentences to embed and store
        collection_name: Name of the reference collection
        overwrite: If True, recreate collection (default: True)
        enable_context: Override global contextualization setting (optional)
    """
    if not sentences:
        print("❌ No sentences provided for embedding.")
        return 0
    
    print(f"📊 Processing {len(sentences)} sentences for reference collection '{collection_name}'...")
    
    # Ensure clean reference collection
    ensure_reference_collection(collection_name, overwrite=overwrite)
    
    # Contextualize sentences
    contextualized_sentences = contextualize_sentences(sentences, enable_context)
    
    # Get embeddings from OpenAI in batch (use contextualized versions)
    print(f"   🤖 Getting embeddings from OpenAI...")
    response = openai.embeddings.create(
        input=contextualized_sentences,
        model="text-embedding-ada-002"
    )
    embeddings = [item.embedding for item in response.data]
    
    # Create points with unique IDs
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                "sentence": sentence,  # Store original sentence
                "contextualized_sentence": contextualized_sentence  # Store contextualized version
            }
        )
        for sentence, contextualized_sentence, embedding in zip(sentences, contextualized_sentences, embeddings)
    ]
    
    # Store in reference collection
    print(f"   💾 Storing {len(points)} embeddings...")
    client.upsert(collection_name=collection_name, points=points)
    
    # Verify storage
    collection_info = client.get_collection(collection_name)
    actual_count = collection_info.points_count
    
    context_status = "with contextualization" if (enable_context if enable_context is not None else ENABLE_CONTEXTUALIZATION) else "without contextualization"
    print(f"   ✅ Stored {actual_count} embeddings in reference collection '{collection_name}' ({context_status})")
    
    if actual_count != len(sentences):
        print(f"   ⚠️  Warning: Expected {len(sentences)} but stored {actual_count}")
    
    return actual_count

if __name__ == "__main__":
    ensure_collection()
    test_sentence = "Our company values innovation and teamwork."
    
    # Test single embedding (with contextualization by default)
    embed_and_store(test_sentence)
    
    # Test bulk embedding
    test_sentences = [
        "We encourage open communication.",
        "Customer satisfaction is our top priority.",
        "We value diversity and inclusion."
    ]
    embed_and_store_bulk(test_sentences)
    
    # Test with contextualization disabled
    print("\n--- Testing without contextualization ---")
    embed_and_store(test_sentence, enable_context=False)
    embed_and_store_bulk(test_sentences, enable_context=False)
    
    print(f"\nContextualization is {'ENABLED' if ENABLE_CONTEXTUALIZATION else 'DISABLED'} by default")
    print(f"Domain context prefix: '{DOMAIN_CONTEXT_PREFIX}'")
    print("Use enable_context=True/False to override the default setting")
