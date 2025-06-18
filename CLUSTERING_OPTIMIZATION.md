# Clustering Quality Assessment System

The quantize_culture project includes a comprehensive two-tier assessment system that evaluates clustering quality from both mathematical and business perspectives.

## 🎯 Assessment Framework

### QUANTITATIVE MEASURES (Mathematical/Statistical)
**Purpose**: Optimize clustering parameters using mathematical metrics for technical excellence

**Key Metrics**:
- **Silhouette Score**: Measures how well points fit their assigned clusters vs. other clusters
- **Davies-Bouldin Index**: Evaluates cluster separation and compactness  
- **Calinski-Harabasz Score**: Assesses cluster density and separation
- **Noise Percentage**: Proportion of points classified as noise (outliers)
- **Cluster Count**: Number of meaningful clusters found (with balanced penalties)

**Implementation**:
- Grid search across 9 different UMAP + HDBSCAN parameter combinations
- Automated parameter optimization using composite scoring
- Historical benchmarking and performance tracking
- Mathematical regression detection

### QUALITATIVE MEASURES (Semantic/Cultural)
**Purpose**: Evaluate business relevance and cultural interpretability for practical utility

**Key Metrics**:
- **Semantic Coherence**: Embedding-based similarity within clusters (cosine similarity)
- **Cultural Alignment**: Alignment with organizational culture dimensions (performance, innovation, collaboration, values, quality)
- **Business Interpretability**: LLM-assessed thematic coherence and actionable business value
- **Theme Clarity**: Whether clusters represent clear, actionable cultural insights

**Implementation**:
- OpenAI embedding analysis for semantic coherence
- Research-based cultural dimension mapping (Cameron & Quinn, Hofstede frameworks)
- GPT-3.5 evaluation for business interpretability and theme naming
- Cultural coverage assessment across multiple dimensions

### Combined Assessment Framework
The system intelligently combines both assessment types:
- **40% Quantitative Weight**: Mathematical clustering quality ensures technical soundness
- **60% Qualitative Weight**: Business relevance ensures practical utility
- **Unified Scoring**: Single combined score balancing technical and business excellence
- **Actionable Recommendations**: Specific guidance based on both assessment types

## 🚀 Usage

### Comprehensive Assessment
```python
from app.clustering_optimizer import ClusteringOptimizer

optimizer = ClusteringOptimizer()
results = optimizer.run_comprehensive_assessment(
    embeddings, 
    include_qualitative=True  # Enable semantic/cultural assessment
)
```

### Quick Start - Optimized Workflow
```bash
python optimized_clustering_workflow.py
```

This runs the complete two-tier assessment pipeline:
1. **QUANTITATIVE**: Tests 9 parameter combinations using mathematical metrics
2. **QUALITATIVE**: Evaluates semantic coherence and cultural interpretability  
3. Combines scores with 40%/60% weighting
4. Stores results and updates benchmark history
5. Generates recommendations and visualizations

# Access results
quality_score = results['quality_metrics']['quality_score']
best_method = results['applied_params']['name']
n_clusters = results['quality_metrics']['n_clusters']
```

### Using with Existing DataExtractorAnalyzer
```python
from app.extract import DataExtractorAnalyzer

analyzer = DataExtractorAnalyzer()
results = analyzer.run_optimized_clustering(limit=200)
```

## 📊 Parameter Grid

The system tests these parameter combinations:

| Method | UMAP Neighbors | UMAP Min Dist | HDBSCAN Min Cluster | HDBSCAN Min Samples |
|--------|----------------|---------------|---------------------|---------------------|
| Baseline | 15 | 0.1 | 5 | 3 |
| High_Resolution | 8 | 0.01 | 2 | 1 |
| Aggressive | 5 | 0.0 | 3 | 2 |
| Conservative | 20 | 0.2 | 8 | 4 |
| Balanced_Tight | 12 | 0.05 | 4 | 2 |
| Balanced_Loose | 25 | 0.1 | 6 | 3 |
| Large_Neighborhood | 30 | 0.05 | 3 | 2 |
| Ultra_Fine | 6 | 0.001 | 2 | 1 |
| Moderate | 15 | 0.05 | 5 | 2 |

## 📈 Quality Scoring Formula

```
Quality Score = (
    cluster_score * 0.2 +      # Favor moderate number of clusters
    noise_score * 0.4 +        # Heavily penalize high noise
    silhouette_score * 0.3 +   # Reward good cluster separation
    ch_score * 0.05 +          # Calinski-Harabasz bonus
    db_score * 0.05            # Davies-Bouldin bonus
)
```

## 📋 Example Quality Scores

| Scenario | Clusters | Noise % | Silhouette | Quality Score |
|----------|----------|---------|------------|---------------|
| Perfect | 20 | 0% | 0.8 | 6.8 |
| Good | 15 | 5% | 0.6 | 5.9 |
| Average | 10 | 15% | 0.4 | 4.8 |
| Poor | 3 | 60% | 0.2 | 2.3 |
| Terrible | 1 | 90% | 0.1 | 0.7 |

## 📁 Generated Files

The optimization system generates:
- `optimization_results.png` - Parameter comparison visualization
- `optimized_clustering.png` - Final clustering plot
- `clustering_benchmarks.json` - Historical performance data
- `out_01.json` - Enhanced output with cluster assignments

## 🔍 Demonstration

Run the demo to see the optimization system in action:
```bash
python clustering_demo.py
```

This demonstrates:
- Quality scoring with different scenarios
- Poor vs good parameter comparison
- Benchmark tracking and trend analysis
- Complete optimization workflow

## 🎯 Performance Results

Current best performance:
- **Method**: Ultra_Fine
- **Quality Score**: 7.2/10
- **Clusters**: 57
- **Noise**: 5.5%
- **Silhouette Score**: 0.587

## 🔧 Customization

### Adding New Parameter Sets
Edit `app/clustering_optimizer.py` and modify `_get_parameter_grid()`:

```python
def _get_parameter_grid(self) -> List[Dict]:
    return [
        # ... existing parameters ...
        {"name": "Custom", "umap_neighbors": 10, "umap_min_dist": 0.02, 
         "hdbscan_min_cluster": 3, "hdbscan_min_samples": 2},
    ]
```

### Modifying Quality Scoring
Adjust weights in `_calculate_quality_score()`:

```python
quality_score = (
    cluster_score * 0.3 +      # Increase cluster importance
    noise_score * 0.3 +        # Reduce noise penalty
    silhouette_score * 0.4     # Increase silhouette importance
)
```

## 📊 Integration

The optimization system is now the standard clustering workflow. Every clustering operation:
1. ✅ Tests multiple parameter combinations
2. ✅ Provides quality scores and benchmarking
3. ✅ Maintains historical performance data
4. ✅ Generates comprehensive visualizations
5. ✅ Automatically selects optimal parameters
