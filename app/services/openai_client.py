"""OpenAI service wrapper for embedding generation."""

import openai
from typing import List, Optional
from ..utils.config import config
from ..data.models import Sentence

class OpenAIService:
    """Service for interacting with OpenAI API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize OpenAI service."""
        self.api_key = api_key or config.openai_api_key
        openai.api_key = self.api_key
        self.embedding_model = config.embedding_model
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of embedding values
        """
        try:
            response = openai.Embedding.create(
                input=text,
                model=self.embedding_model
            )
            return response['data'][0]['embedding']
        except Exception as e:
            raise Exception(f"Failed to generate embedding: {str(e)}")
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts to embed
            
        Returns:
            List of embedding lists
        """
        try:
            response = openai.Embedding.create(
                input=texts,
                model=self.embedding_model
            )
            return [item['embedding'] for item in response['data']]
        except Exception as e:
            raise Exception(f"Failed to generate embeddings: {str(e)}")
    
    def get_embeddings_for_sentences(self, sentences: List[Sentence]) -> List[Sentence]:
        """
        Generate embeddings for sentence objects.
        
        Args:
            sentences: List of Sentence objects
            
        Returns:
            List of Sentence objects with embeddings populated
        """
        # Extract texts (use contextualized text if available)
        texts = []
        for sentence in sentences:
            text = sentence.contextualized_text if sentence.contextualized_text else sentence.text
            texts.append(text)
        
        # Generate embeddings
        embeddings = self.get_embeddings(texts)
        
        # Update sentence objects
        for sentence, embedding in zip(sentences, embeddings):
            sentence.embedding = embedding
        
        return sentences
