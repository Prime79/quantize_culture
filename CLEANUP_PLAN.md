# Repository Cleanup Plan

## PHASE 1: Remove Temporary/Generated Files ✅

### A. Quadrant DB Files (Already in .gitignore but exist locally)
- Remove entire `quadrant/` directory (vector DB data files)

### B. Temporary JSON Files
- `data_01.json` / `data_01_contextualized.json`
- `out_01.json`
- `extended_dl_sentences.json` / `extended_dl_sentences_contextualized.json`
- `clustering_benchmarks.json`
- `demo_analysis_results.json`

### C. Generated PNG Files
- `company_culture_clusters.png`
- `improved_clustering.png`
- `optimization_results.png`
- `optimized_clustering.png`
- `parameter_comparison.png`
- `sample_comparison.png`
- `sample_summary.png`
- `sample_umap_2d.png`
- `umap_2d_clusters.png`

### D. Cache and Build Files
- `__pycache__/` directories
- `.pytest_cache/`

## PHASE 2: Remove Experimental/Demo Python Files ✅

### Clearly Experimental/Demo Files:
- `benchmark_comparison.py`
- `clustering_demo.py`
- `create_output.py`
- `create_enhanced_output.py`
- `create_enhanced_output_simple.py`
- `create_contextualized_extended.py`
- `complete_scoring_demo.py`
- `full_scoring_demo.py`
- `quick_scoring_demo.py`
- `debug_*.py` files
- `demo_*.py` files
- `examples_extract.py`
- `fixed_extract.py`
- `test_*.py` files (not in tests/ directory)
- `plot_*.py` files
- `simple_plot.py`
- `generate_plots.py`
- `improve_clustering.py`
- `optimized_clustering_workflow.py`
- `run_full_workflow.py`

### Keep for Review:
- `app/` directory (core functionality)
- `tests/` directory (our BDD test suite)
- Configuration files (.env, .gitignore, requirements.txt, etc.)
- Documentation files (README.md, *.md)

## PHASE 3: Consolidate and Refactor Core App

### Core Files to Review and Refactor:
- `app/main.py` - Main entry point
- `app/extract.py` - Data extraction functionality
- `app/embed_and_store.py` - Embedding and storage
- `app/clustering_optimizer.py` - Clustering logic
- `app/qualitative_assessment.py` - Assessment logic
- `app/load_data.py` - Data loading utilities
- `app/utils.py` - Utility functions

## PHASE 4: Clean Architecture

### Proposed Clean Structure:
```
quantize_culture/
├── app/
│   ├── core/           # Core business logic
│   ├── data/           # Data handling
│   ├── models/         # Data models
│   ├── services/       # External services (OpenAI, Qdrant)
│   └── utils/          # Utilities
├── tests/              # BDD test suite (already clean)
├── docs/               # Documentation
├── config/             # Configuration files
└── scripts/            # Utility scripts
```

## PHASE 5: Final Validation

- Run BDD test suite after each phase
- Update documentation
- Verify core functionality works
- Clean up dependencies in requirements.txt
