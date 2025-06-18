# Clustering Optimization System

The quantize_culture project now includes an advanced clustering optimization system that automatically finds the best parameters and benchmarks results against historical performance.

## 🎯 Features

### Automatic Parameter Optimization
- **Grid Search**: Tests 9 different parameter combinations for UMAP + HDBSCAN
- **Quality Scoring**: Comprehensive scoring system combining multiple metrics
- **Best Selection**: Automatically selects optimal parameters based on quality score

### Quality Metrics
The system evaluates clustering quality using:
- **Number of Clusters**: Balanced penalty for too few/many clusters
- **Noise Percentage**: Lower noise = higher quality
- **Silhouette Score**: Cluster separation quality
- **Calinski-Harabasz Score**: Cluster density and separation
- **Davies-Bouldin Score**: Average similarity between clusters

### Benchmarking & History
- **Historical Tracking**: Maintains performance history in `clustering_benchmarks.json`
- **Best Ever Tracking**: Records the best quality score achieved
- **Trend Analysis**: Shows performance trends over multiple runs
- **Regression Detection**: Alerts when performance drops significantly

## 🚀 Usage

### Quick Start - Optimized Workflow
```bash
python optimized_clustering_workflow.py
```

This runs the complete optimization pipeline:
1. Extracts data from Qdrant
2. Tests 9 parameter combinations
3. Selects best parameters
4. Applies clustering and stores results
5. Updates benchmark history
6. Generates visualizations

### Programmatic Usage
```python
from app.clustering_optimizer import EnhancedDataExtractorAnalyzer

# Initialize enhanced analyzer
analyzer = EnhancedDataExtractorAnalyzer()

# Run optimization and clustering
results = analyzer.optimize_and_cluster(limit=200)

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
