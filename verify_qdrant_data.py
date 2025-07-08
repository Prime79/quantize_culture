import polars as pl
from qdrant_client import QdrantClient
from pathlib import Path

# Initialize Qdrant client to use the file-based database
script_dir = Path(__file__).parent
qdrant_path = script_dir / "target" / "qdrant_db"
qdrant_client = QdrantClient(path=str(qdrant_path))

# Fetch data from Qdrant
points, _ = qdrant_client.scroll(
    collection_name="target_test",
    limit=10,  # Fetch a few points to verify
    with_payload=True,
    with_vectors=False
)

# Print the payload to verify the informant field
for point in points:
    print(point.payload)
