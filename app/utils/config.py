"""Configuration management for the Digital Leadership Assessment pipeline."""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Application configuration."""
    
    def __init__(self):
        # OpenAI Configuration
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Qdrant Configuration
        self.qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        self.qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
        self.default_collection = os.getenv("QDRANT_COLLECTION", "company_culture_embeddings")
        
        # Contextualization Settings
        self.domain_context_prefix = os.getenv(
            "DOMAIN_CONTEXT_PREFIX", 
            "This is a sentence related to digital rights, privacy, technology, and online culture:"
        )
        self.enable_contextualization = os.getenv("ENABLE_CONTEXTUALIZATION", "true").lower() == "true"
        
        # Embedding Settings
        self.embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
        self.embedding_dimensions = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))
        
        # Clustering Settings
        self.default_min_cluster_size = int(os.getenv("MIN_CLUSTER_SIZE", "5"))
        self.default_min_samples = int(os.getenv("MIN_SAMPLES", "3"))
        self.default_n_neighbors = int(os.getenv("UMAP_N_NEIGHBORS", "15"))
        self.default_min_dist = float(os.getenv("UMAP_MIN_DIST", "0.1"))
    
    def validate(self) -> bool:
        """Validate that all required configuration is present."""
        required_fields = [
            'openai_api_key',
            'qdrant_host',
            'qdrant_port'
        ]
        
        for field in required_fields:
            if not getattr(self, field):
                raise ValueError(f"Required configuration field '{field}' is missing")
        
        return True

# Global configuration instance
config = Config()
