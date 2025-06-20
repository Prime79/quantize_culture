# Repository Refactoring - Phase 3: Architecture Improvements

## Current State Analysis

### Current Files:
- `app/main.py` - Minimal entry point (4 lines)
- `app/extract.py` - Large monolithic file (466 lines) - data extraction, UMAP, HDBSCAN, plotting
- `app/embed_and_store.py` - Embedding and storage logic (241 lines)
- `app/qualitative_assessment.py` - Cultural/semantic assessment (511 lines)
- `app/qualitative_assessment_run.py` - Assessment runner
- `app/clustering_optimizer.py` - Clustering optimization
- `app/load_data.py` - Data loading utilities
- `app/utils.py` - General utilities

### Issues Identified:
1. **Monolithic files** - `extract.py` does too many things
2. **Mixed concerns** - Plotting mixed with data processing
3. **No clear entry points** - `main.py` is just a placeholder
4. **Duplicate functionality** - Some overlap between files
5. **Poor separation** - Business logic mixed with infrastructure

## Proposed Clean Architecture

```
app/
├── __init__.py
├── main.py                    # Main CLI interface
├── core/                      # Core business logic
│   ├── __init__.py
│   ├── pipeline.py           # Main DL assessment pipeline
│   ├── contextualization.py  # Context enhancement
│   ├── embedding.py          # Embedding generation  
│   ├── clustering.py         # Clustering algorithms
│   └── assessment.py         # Qualitative assessment
├── data/                      # Data handling
│   ├── __init__.py
│   ├── loader.py             # Data loading
│   ├── extractor.py          # Data extraction
│   └── models.py             # Data models/schemas
├── services/                  # External services
│   ├── __init__.py
│   ├── openai_client.py      # OpenAI API wrapper
│   ├── qdrant_client.py      # Qdrant vector DB wrapper
│   └── storage.py            # Storage operations
├── analysis/                  # Analysis and optimization
│   ├── __init__.py
│   ├── optimizer.py          # Clustering optimization
│   ├── metrics.py            # Quality metrics
│   └── visualization.py     # Plotting and visualization
└── utils/                     # Utilities
    ├── __init__.py
    ├── config.py             # Configuration management
    └── helpers.py            # Helper functions
```

## Benefits of This Structure:
1. **Single Responsibility** - Each module has one clear purpose
2. **Dependency Injection** - Services can be easily mocked for testing
3. **Testable Architecture** - Aligns perfectly with our BDD test suite
4. **Maintainable** - Changes are isolated to specific areas
5. **Extensible** - Easy to add new features without breaking existing code
