# quantize_culture

## Project Overview

This project is a web application designed to estimate the dominant logic of company culture by analyzing typical behavioral statements. The app will be deployed to Google Cloud Platform (GCP).

## Key Features & Plan

- **Sentence Embedding:**
  - Uses OpenAI Embedding API to embed behavioral sentences.
- **Data Storage:**
  - Stores embeddings in a Quadrant DB.
- **Dimensionality Reduction:**
  - Applies UMAP to downscale embeddings to a 3D space for visualization and analysis.
- **Clustering:**
  - Uses HDBSCAN to find clusters in the 3D embedding space.
- **Dominant Logic Estimation:**
  - Estimates the dominant logic of company culture from the identified clusters.
- **Containerization:**
  - All components run in containers.
  - Persistent data and content are stored in this project folder.
- **Deployment:**
  - The app will be deployed to GCP for scalability and reliability.

## Quadrant DB Container Setup

### Step 1: Quadrant DB Containerization
- Created a `quadrant` directory with a `Dockerfile` to run the official Quadrant DB image.
- Exposes port 8080 and mounts persistent storage to `./quadrant/data`.
- Added a `docker-compose.yml` to manage the Quadrant DB container easily.

**How to start Quadrant DB locally:**
```sh
docker-compose up -d quadrantdb
```
This will build and start the Quadrant DB container with persistent storage.

### Step 1 Update: Qdrant Vector DB Containerization
- Switched to the official Qdrant vector database image (`qdrant/qdrant`).
- Updated the `Dockerfile` and `docker-compose.yml` for correct ports and persistent storage.
- Qdrant runs on port 6333 and stores data in `./quadrant/data`.

**How to start Qdrant locally:**
```sh
docker-compose up -d qdrant
```
Qdrant will be available at `http://localhost:6333`.

### Step 2: Python Environment & Qdrant Test
- Created a Python virtual environment for the project (`.venv`).
- Installed the `qdrant-client` package for Python.
- Added `test.py` to verify Qdrant connectivity from Python.

**How to test Qdrant connection:**
```sh
.venv/bin/python test.py
```
This script prints the Qdrant version if the connection is successful.

## Next Steps
1. Define the application architecture and technology stack.
2. Set up the web app framework and containerization.
3. Integrate OpenAI Embedding API.
4. Implement Quadrant DB storage.
5. Add UMAP and HDBSCAN processing.
6. Develop logic estimation and visualization.
7. Prepare for GCP deployment.

---

Feel free to update this README as the project evolves!
