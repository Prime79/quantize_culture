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

### Step 3: Secure OpenAI API Key Storage
- Added a `.env` file template for the OpenAI API key.
- Updated `.gitignore` to exclude `.env` from version control.
- Installed `python-dotenv` and updated `test.py` to load and check the API key from the environment.

**How to use:**
1. Copy your OpenAI API key into the `.env` file:
   ```
   OPENAI_API_KEY=sk-...
   ```
2. Run Python scripts as usual; the key will be loaded automatically.

### Step 4: App Container Setup
- Created an `app` folder for the main application code.
- Added a lightweight `Dockerfile` for the app, installing only required system and Python dependencies.
- Created a `requirements.txt` with: `openai`, `qdrant-client`, `umap-learn`, `hdbscan`, `jax`, and `python-dotenv`.
- Added a placeholder `main.py` as the app entry point.

**How to build and run the app container:**
```sh
cd app
# Build the container
docker build -t quantize_culture_app .
# Run the container
# (Mount .env and other needed files as needed)
docker run --rm quantize_culture_app
```

### Step 5: End-to-End Embedding Test
- Added a test to embed a sentence using the OpenAI API, store it in Qdrant, verify its presence, and remove it after the test.
- The test uses a valid UUID for the Qdrant point ID and the correct deletion selector.

**How to run the embedding test:**
```sh
.venv/bin/python test.py
```
This will:
- Embed a test sentence
- Store it in the `embeddings_test` collection in Qdrant
- Verify the sentence is present
- Remove the test embedding from Qdrant

### Step 6: Bulk Embedding Test
- Added a function to embed and store a list of sentences in Qdrant in bulk, assigning a UUID to each point.
- Added a test to verify bulk embedding, storage, and removal of multiple sentences.

**How to run the bulk embedding test:**
```sh
.venv/bin/python test.py
```
This will:
- Embed and store multiple test sentences in the `embeddings_test_bulk` collection
- Verify all sentences are present
- Remove the test embeddings from Qdrant

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
