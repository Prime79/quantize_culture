#!/usr/bin/env python3
"""
Ray-based embedding processor for Xylem interview data.
Processes passages in parallel using Ray remote functions.
"""

import polars as pl
import openai
import os
import ray
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import NamedTuple, List
import uuid
import time

# Load environment variables
load_dotenv()

class PassageInput(NamedTuple):
    """Input data structure for passages."""
    passage: str
    dominant_logic: str
    informant: str

class EmbeddingResult(NamedTuple):
    """Result data structure with embeddings."""
    passage: str
    dominant_logic: str
    informant: str
    embedding: List[float]

@ray.remote
def embed_passage(passage_input: PassageInput) -> EmbeddingResult:
    """
    Ray remote function to embed a single passage using OpenAI API.
    
    Args:
        passage_input: NamedTuple with passage, dominant_logic, and informant
        
    Returns:
        EmbeddingResult: NamedTuple with passage, dominant_logic, informant, and embedding
    """
    try:
        # Create OpenAI client
        client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Create embedding
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=passage_input.passage
        )
        
        embedding = response.data[0].embedding
        
        return EmbeddingResult(
            passage=passage_input.passage,
            dominant_logic=passage_input.dominant_logic,
            informant=passage_input.informant,
            embedding=embedding
        )
        
    except Exception as e:
        print(f"❌ Error embedding passage: {e}")
        # Return empty embedding on error
        return EmbeddingResult(
            passage=passage_input.passage,
            dominant_logic=passage_input.dominant_logic,
            informant=passage_input.informant,
            embedding=[]
        )

def read_parquet_to_named_tuples() -> List[PassageInput]:
    """
    Step 1 & 2: Read parquet file and collect all lines to a list of named tuples.
    
    Returns:
        List[PassageInput]: List of named tuples with passage, dominant_logic, and informant
    """
    print("📁 Step 1: Reading parquet file...")
    
    script_dir = Path(__file__).parent
    parquet_path = script_dir / "imported_data.parquet"
    
    if not parquet_path.exists():
        raise FileNotFoundError("imported_data.parquet not found. Run import_excel_to_polars.py first")
    
    df = pl.read_parquet(parquet_path)
    print(f"✅ Loaded data: {df.shape}")
    
    print("📋 Step 2: Collecting passages to named tuples...")
    
    # Filter valid passages (non-null and meaningful length)
    valid_df = df.filter(
        pl.col('passage').is_not_null() &
        (pl.col('passage').str.len_chars() > 10)
    )
    
    print(f"📊 Found {valid_df.height} valid passages out of {df.height} total rows")
    
    # Convert to list of named tuples
    passage_list = [
        PassageInput(
            passage=row['passage'],
            dominant_logic=row['dominant_logic'],
            informant=row['informant']
        )
        for row in valid_df.to_dicts()
    ]
    
    print(f"✅ Converted {len(passage_list)} passages to named tuples")
    return passage_list

def process_with_ray(passage_list: List[PassageInput]) -> List[EmbeddingResult]:
    """
    Step 3: Process all passages using Ray remote functions.
    
    Args:
        passage_list: List of PassageInput named tuples
        
    Returns:
        List[EmbeddingResult]: List of results with embeddings
    """
    print(f"🚀 Step 3: Processing {len(passage_list)} passages with Ray...")
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init()
        print("✅ Ray initialized")
    
    # Submit all tasks to Ray
    print("📤 Submitting all tasks to Ray remote functions...")
    ray_futures = []
    for i, passage_input in enumerate(passage_list):
        future = embed_passage.remote(passage_input)
        ray_futures.append(future)
        if (i + 1) % 50 == 0:  # Progress update every 50 submissions
            print(f"   📤 Submitted {i + 1}/{len(passage_list)} tasks...")
    
    print(f"✅ Submitted {len(ray_futures)} tasks to Ray")
    
    # Wait for all results
    print("⏳ Waiting for all Ray remote functions to complete...")
    start_time = time.time()
    
    results = ray.get(ray_futures)
    
    end_time = time.time()
    print(f"✅ All Ray tasks completed in {end_time - start_time:.2f} seconds")
    
    # Filter out failed embeddings (empty embedding lists)
    successful_results = [r for r in results if r.embedding]
    failed_count = len(results) - len(successful_results)
    
    if failed_count > 0:
        print(f"⚠️  {failed_count} embeddings failed, {len(successful_results)} succeeded")
    else:
        print(f"🎉 All {len(successful_results)} embeddings succeeded!")
    
    return successful_results

def write_to_qdrant(embedding_results: List[EmbeddingResult]) -> None:
    """
    Step 4: Write all embedding results to Qdrant database.
    
    Args:
        embedding_results: List of EmbeddingResult named tuples
    """
    print("💾 Step 4: Writing results to Qdrant database...")
    
    collection_name = "target_test"
    
    # Initialize Qdrant client to use a file-based database
    script_dir = Path(__file__).parent
    qdrant_path = script_dir / "qdrant_db"
    qdrant_client = QdrantClient(path=str(qdrant_path))
    
    # Create collection if it doesn't exist
    try:
        qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
        print(f"✅ Collection '{collection_name}' created successfully.")
    except Exception as e:
        print(f"⚠️ Collection '{collection_name}' already exists or error: {e}")

    # Prepare points for upsert
    points_to_upsert = []
    for result in embedding_results:
        if result.embedding:
            points_to_upsert.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=result.embedding,
                    payload={
                        "passage": result.passage,
                        "dominant_logic": result.dominant_logic,
                        "informant": result.informant
                    }
                )
            )
            
    if not points_to_upsert:
        print("🤷 No valid embeddings to store.")
        return

    # Upsert points to Qdrant
    try:
        qdrant_client.upsert(
            collection_name=collection_name,
            points=points_to_upsert,
            wait=True
        )
        print(f"✅ Successfully stored {len(points_to_upsert)} points in Qdrant.")
        
        # Verify count
        count_result = qdrant_client.count(collection_name=collection_name, exact=True)
        print(f"   📊 Collection count: {count_result.count}")

    except Exception as e:
        print(f"❌ Error storing embeddings in Qdrant: {e}")
        raise

def main():
    """Main function to orchestrate the entire process."""
    print("🎯 Ray-based Embedding Processor for Xylem Interview Data")
    print("=" * 60)
    
    # Check environment setup
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY not found in environment")
        return
    else:
        print("✅ OpenAI API key found")
    
    try:
        # Step 1 & 2: Read parquet and collect to named tuples
        passage_list = read_parquet_to_named_tuples()
        
        if not passage_list:
            print("❌ No valid passages found")
            return
        
        # Step 3: Process with Ray remote functions
        embedding_results = process_with_ray(passage_list)
        
        if not embedding_results:
            print("❌ No successful embeddings")
            return
        
        # Step 4: Write to Qdrant
        write_to_qdrant(embedding_results)
        
        print(f"\n🎉 Process completed successfully!")
        print(f"📊 Processed {len(passage_list)} passages")
        print(f"✅ Successfully embedded {len(embedding_results)} passages")
        print(f"💾 Stored in Qdrant collection 'target_test'")
        
    except Exception as e:
        print(f"❌ Error in main process: {e}")
        raise
    
    finally:
        # Shutdown Ray
        if ray.is_initialized():
            ray.shutdown()
            print("🔄 Ray shutdown complete")

if __name__ == "__main__":
    main()
