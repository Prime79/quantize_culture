import pandas as pd
import umap
import hdbscan
from qdrant_client import QdrantClient
from qdrant_client.models import PointIdsList
from pathlib import Path

def run_hdbscan_and_update():
    """Run HDBSCAN on 3D UMAP data and update Qdrant with cluster labels."""
    print("🚀 Starting HDBSCAN clustering and Qdrant update...")

    # Connect to Qdrant
    script_dir = Path(__file__).parent
    qdrant_path = script_dir / "qdrant_db"
    qdrant_client = QdrantClient(path=str(qdrant_path))
    print("✅ Connected to Qdrant database.")

    # Fetch all data
    points, _ = qdrant_client.scroll(
        collection_name="target_test",
        with_payload=True,
        with_vectors=True,
        limit=1000
    )
    print(f"✅ Fetched {len(points)} points from Qdrant.")

    # Prepare data
    embeddings = [point.vector for point in points]
    point_ids = [point.id for point in points]

    # Perform 3D UMAP reduction
    reducer = umap.UMAP(n_components=3, random_state=42)
    embedding_3d = reducer.fit_transform(embeddings)
    print("✅ UMAP reduction to 3D complete.")

    # Perform HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)
    cluster_labels = clusterer.fit_predict(embedding_3d)
    print(f"✅ HDBSCAN clustering complete. Found {len(set(cluster_labels)) - 1} clusters.")

    # Update Qdrant with cluster labels one by one
    print("🔄 Updating Qdrant with HDBSCAN cluster labels...")
    for i, point_id in enumerate(point_ids):
        cluster_label = int(cluster_labels[i])
        qdrant_client.set_payload(
            collection_name="target_test",
            payload={"hdbscan_cluster": cluster_label},
            points=[point_id],
            wait=True
        )
    print("✅ Updated Qdrant with HDBSCAN cluster labels.")

    # Verification
    verify_points, _ = qdrant_client.scroll(
        collection_name="target_test", limit=5, with_payload=True
    )
    print("\n🔍 Verification:")
    for point in verify_points:
        print(point.payload)

if __name__ == "__main__":
    run_hdbscan_and_update()
