# quantize_culture

## Project Overview

This project is a comprehensive platform for analyzing company culture through behavioral statements using advanced embedding, clustering, and assessment techniques. The system provides both quantitative (mathematical/statistical) and qualitative (semantic/cultural) evaluation of organizational culture patterns.

## Key Features & Architecture

- **Sentence Embedding:**
  - Uses OpenAI Embedding API (text-embedding-ada-002) to embed behavioral sentences
  - Supports bulk embedding operations for efficient processing
- **Vector Storage:**
  - Stores embeddings in Qdrant vector database with persistent storage
  - Robust collection management with reference collection overwriting
- **Clustering Pipeline:**
  - UMAP dimensionality reduction for visualization and analysis
  - HDBSCAN density-based clustering with parameter optimization
  - Maximum cluster limit enforcement (configurable, default: 50 clusters)
- **Digital Leadership Inference:**
  - **Command-Line Interface**: Direct sentence classification via CLI
  - **Enhanced Inference Engine**: Multi-factor confidence assessment with semantic analysis
  - **Dominant Logic Detection**: Identifies Digital Leadership categories, subcategories, and archetypes
  - **Cluster Analysis**: Complete cluster membership analysis with DL metadata
- **Dual Assessment System:**
  - **Quantitative**: Mathematical clustering quality (silhouette, noise percentage, parameter optimization)
  - **Qualitative**: Semantic coherence, cultural alignment, business interpretability
- **Full Workflow Automation:**
  - End-to-end pipeline from JSON input to clustered output with comprehensive reporting
- **Visualization & Reporting:**
  - Automated plot generation for cluster visualization
  - Markdown reports with verbal evaluation and recommendations
- **Containerization:**
  - All components run in containers with persistent data storage
- **Comprehensive Testing:**
  - BDD test framework with 10+ test scenarios
  - Test-driven development approach
  - Continuous integration with pytest

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

## Digital Leadership Inference CLI

### Overview

The Digital Leadership Inference CLI provides a command-line interface for real-time classification of sentences into Digital Leadership archetypes. It leverages the embedded reference database to perform semantic similarity matching and provides comprehensive analysis of dominant logic patterns.

### Key Features

- **🎯 Real-time Classification**: Instant DL archetype inference for any input sentence
- **🧠 Dominant Logic Analysis**: Identifies most common DL categories, subcategories, and archetypes
- **📊 Cluster Analysis**: Complete cluster membership analysis with metadata
- **🔍 Multi-format Output**: Human-readable and JSON formats
- **⚡ Performance Optimized**: Sub-30 second response times
- **🛡️ Error Resilient**: Comprehensive error handling and validation

### Quick Start

```bash
# Basic sentence classification
python inference.py -s "order is the key for success" -c extended_contextualized_collection

# Detailed analysis with cluster members and dominant logic
python inference.py -s "order is the key for success" -c extended_contextualized_collection --verbose

# JSON output for programmatic use
python inference.py -s "order is the key for success" -c extended_contextualized_collection --format json

# Get help and see all options
python inference.py --help
```

### Example Output

**Human-Readable Format:**
```
📝 Sentence: "order is the key for success"
🎯 Digital Leadership Classification: cluster_46
📊 Cluster ID: 46
🔍 Similarity Score: 0.9281
✅ Confidence Level: AMBIGUOUS

🧠 DOMINANT LOGIC ANALYSIS:
   📂 Primary Category: Strategic Planning (5/8 sentences)
   📋 Primary Subcategory: Execution Focus (3/8 sentences)
   🎭 Primary Archetype: Results-Oriented (4/8 sentences)

📚 ALL CLUSTER MEMBERS (8 total):
    1. Execution matters less than outcome. [Strategic Planning → Execution Focus → Results-Oriented]
    2. Planning is crucial for success [Strategic Planning → Process Design → Systematic]
    ...
```

**JSON Format:**
```json
{
  "sentence": "order is the key for success",
  "cluster_id": 46,
  "similarity_score": 0.9281,
  "classification": "cluster_46",
  "confidence_level": "AMBIGUOUS",
  "dominant_logic": {
    "most_common_category": ["Strategic Planning", 5],
    "most_common_subcategory": ["Execution Focus", 3],
    "most_common_archetype": ["Results-Oriented", 4]
  },
  "cluster_size": 8
}
```

### Advanced Usage

#### Batch Processing
```bash
# Process multiple sentences
for sentence in "fail fast learn faster" "innovation drives success" "teamwork matters most"; do
    python inference.py -s "$sentence" -c extended_contextualized_collection --format json
done
```

#### Integration with Scripts
```python
import subprocess
import json

result = subprocess.run([
    'python', 'inference.py', 
    '-s', 'your sentence here',
    '-c', 'extended_contextualized_collection',
    '--format', 'json'
], capture_output=True, text=True)

if result.returncode == 0:
    data = json.loads(result.stdout)
    print(f"Classification: {data['classification']}")
    print(f"Confidence: {data['confidence_level']}")
```

### Command Reference

| Option | Short | Description | Example |
|--------|-------|-------------|---------|
| `--sentence` | `-s` | Sentence to classify (required) | `-s "innovation is key"` |
| `--collection` | `-c` | Qdrant collection name (required) | `-c extended_contextualized_collection` |
| `--format` | | Output format: `human` or `json` | `--format json` |
| `--verbose` | `-v` | Include detailed cluster analysis | `-v` |
| `--help` | `-h` | Show help message | `-h` |
| `--version` | | Show version information | `--version` |

### Testing

The CLI includes comprehensive BDD test coverage:

```bash
# Run all BDD tests (including CLI tests)
python run_bdd_tests.py

# Run only CLI tests
python run_bdd_tests.py cli

# Run integration tests
python test_inference_cli.py
```

**Test Coverage:**
- ✅ 10 BDD test scenarios
- ✅ 100% CLI argument coverage
- ✅ Error handling validation
- ✅ Performance testing (sub-30s)
- ✅ Integration with inference engine

### Requirements

- OpenAI API key (set `OPENAI_API_KEY` environment variable)
- Qdrant database running with populated collection
- Python dependencies from `requirements.txt`

---

## Data Analysis and Clustering

The `app/extract.py` script provides comprehensive analysis capabilities for the embedded company culture data:

### Four Main Functionalities

1. **Data Extraction**: Extract all or specific data from the Qdrant database
2. **UMAP Dimensionality Reduction**: Reduce high-dimensional embeddings to 2D or 3D
3. **HDBSCAN Clustering**: Perform density-based clustering on reduced data
4. **Database Storage**: Store cluster labels back to the original embeddings

### Usage Examples

#### Quick Demo

```bash
# Run the full pipeline demo
python demo_extract.py

# Run individual function examples
python examples_extract.py
```

#### Programmatic Usage

```python
from app.extract import DataExtractorAnalyzer

# Initialize analyzer
analyzer = DataExtractorAnalyzer()

# 1. Extract data (all or filtered)
data = analyzer.extract_data(limit=100)  # or None for all data

# 2. Reduce dimensions
reduced = analyzer.reduce_dimensions(n_components=2, n_neighbors=15)

# 3. Cluster the data
clusters = analyzer.cluster_data(min_cluster_size=5)

# 4. Store results back to database
analyzer.store_clusters_to_database()

# Get cluster summary
summary = analyzer.get_cluster_summary()
print(summary)
```

#### Advanced Options

**Filtered Data Extraction:**

```python
# Extract data with specific filters
filters = {"category": "Leadership", "rating_min": 3}
data = analyzer.extract_data(filter_conditions=filters)
```

**3D UMAP Reduction:**

```python
reduced_3d = analyzer.reduce_dimensions(
    n_components=3,
    n_neighbors=10,
    min_dist=0.05,
    metric='cosine'
)
```

**Custom Clustering:**

```python
clusters = analyzer.cluster_data(
    min_cluster_size=8,
    min_samples=4,
    cluster_selection_epsilon=0.1
)
```

### Output Files

- `demo_analysis_results.json`: Full analysis results from demo
- `culture_analysis_results.json`: Results from main analysis

## Full Workflow Pipeline

### Complete End-to-End Processing

The project includes a comprehensive workflow script (`run_full_workflow.py`) that processes company culture data from JSON input to fully analyzed, clustered, and documented results.

**Features:**
- **Input**: JSON file with company culture sentences
- **Processing**: Embedding, optimization, clustering, assessment
- **Output**: Clustered database, visualizations, comprehensive reports

**Usage:**
```python
from run_full_workflow import run_full_workflow

# Run complete pipeline
results = run_full_workflow(
    input_json="your_sentences.json",
    collection_name="your_collection",
    output_dir="reports"
)
```

**What it does:**
1. **Loads sentences** from JSON file (structured by categories/subcategories)
2. **Embeds and stores** all sentences in Qdrant with collection management
3. **Optimizes clustering** parameters across 9 different methods
4. **Applies best clustering** with maximum 50-cluster constraint
5. **Runs comprehensive assessment** (quantitative + qualitative evaluation)
6. **Generates visualizations** (optimization plots, cluster plots)
7. **Creates reports** (Markdown with metrics and verbal evaluation)
8. **Stores cluster IDs** back to the reference database

**Example Results:**
- **43 clusters** from 600 sentences (Fine_Controlled method)
- **16.2% noise** points with **0.534 silhouette** score
- **Quantitative score: 6.3** with full qualitative assessment
- Complete documentation in `reports/clustering_report_[collection]_[timestamp].md`

## Clustering Quality Assessment

The project implements a comprehensive two-tier assessment system to evaluate clustering quality:

### QUANTITATIVE MEASURES (Mathematical/Statistical)

**Purpose**: Optimize clustering parameters using mathematical metrics

**Metrics**:
- **Silhouette Score**: Measures how well points fit their assigned clusters vs. other clusters
- **Davies-Bouldin Index**: Evaluates cluster separation and compactness
- **Noise Percentage**: Proportion of points classified as noise (outliers)
- **Cluster Count**: Number of meaningful clusters found (max 50 enforced)

**Implementation**: `app/clustering_optimizer.py`
- Grid search across UMAP and HDBSCAN parameters
- Automated parameter optimization with cluster limit enforcement
- Benchmarking against previous runs
- Mathematical scoring and ranking

### QUALITATIVE MEASURES (Semantic/Cultural)

**Purpose**: Evaluate business relevance and cultural interpretability

**Metrics**:
- **Semantic Coherence**: How semantically similar sentences are within clusters (using embedding similarity)
- **Cultural Alignment**: Alignment with organizational culture dimensions (performance, innovation, collaboration, etc.)
- **Business Interpretability**: LLM-assessed coherence and actionable business value
- **Theme Clarity**: Whether clusters represent clear, actionable cultural themes

**Implementation**: `app/qualitative_assessment.py`
- OpenAI embedding-based semantic analysis
- Cultural dimension mapping using research-based frameworks
- LLM evaluation for business interpretability
- Thematic analysis and naming

### Combined Assessment

The system combines both assessment types with weighted scoring:
- **40% Quantitative**: Mathematical clustering quality
- **60% Qualitative**: Business relevance and cultural meaning

**Usage**:
```python
# Run comprehensive assessment
from app.clustering_optimizer import EnhancedDataExtractorAnalyzer
analyzer = EnhancedDataExtractorAnalyzer(collection_name="your_collection")
results = analyzer.run_comprehensive_assessment(embeddings, include_qualitative=True)
```

**Benefits**:
- Ensures clusters are both mathematically sound AND business-relevant
- Provides actionable recommendations for improvement
- Tracks quality improvements over time
- Balances technical optimization with practical utility

### Dependencies

The analysis functionality requires additional packages:

- `pandas`: Data manipulation
- `numpy`: Numerical operations
- `scikit-learn`: Preprocessing and utilities
- `umap-learn`: UMAP dimensionality reduction
- `hdbscan`: HDBSCAN clustering

All dependencies are listed in `app/requirements.txt`.

## Next Steps

1. ✅ **Containerized Infrastructure**: Qdrant vector database with persistent storage
2. ✅ **Embedding Pipeline**: OpenAI API integration with bulk processing
3. ✅ **Clustering System**: UMAP + HDBSCAN with parameter optimization
4. ✅ **Quality Assessment**: Dual quantitative/qualitative evaluation system
5. ✅ **Full Workflow**: End-to-end pipeline from JSON to analyzed clusters
6. ✅ **Reporting & Visualization**: Automated plots and markdown reports
7. ✅ **Digital Leadership CLI**: Command-line interface for real-time inference
8. ✅ **Comprehensive Testing**: BDD test framework with test-driven development
9. 🔄 **Web Interface**: Develop user-friendly web application
10. 🔄 **GCP Deployment**: Cloud deployment for scalability
11. 🔄 **API Endpoints**: RESTful API for integration capabilities

## Quick Start

1. **Start Qdrant database:**
   ```bash
   docker-compose up -d qdrant
   ```

2. **Set up environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   pip install -r app/requirements.txt
   ```

3. **Configure API key:**
   ```bash
   echo "OPENAI_API_KEY=your_key_here" > .env
   ```

4. **Run full pipeline:**
   ```bash
   python run_full_workflow.py
   ```

5. **Use Digital Leadership Inference CLI:**
   ```bash
   # Classify any sentence
   python inference.py -s "order is the key for success" -c extended_contextualized_collection
   
   # Get detailed analysis
   python inference.py -s "fail fast learn faster" -c extended_contextualized_collection --verbose
   
   # Run BDD tests
   python run_bdd_tests.py
   ```

This will process the default dataset (`extended_dl_sentences.json`) and generate comprehensive results in the `reports/` directory. The CLI allows for real-time Digital Leadership archetype classification and analysis.

---

Feel free to update this README as the project evolves!
