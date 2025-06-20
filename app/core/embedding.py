"""Embedding generation service for the Digital Leadership Assessment pipeline."""

from typing import List, Optional
from ..data.models import Sentence
from ..services.openai_client import OpenAIService

class EmbeddingService:
    """Service for generating and managing embeddings."""
    
    def __init__(self, openai_service: Optional[OpenAIService] = None):
        """
        Initialize embedding service.
        
        Args:
            openai_service: OpenAI service instance (creates new if None)
        """
        self.openai_service = openai_service or OpenAIService()
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        return self.openai_service.get_embedding(text)
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of embedding vectors
        """
        return self.openai_service.get_embeddings(texts)
    
    def embed_sentence(self, sentence: Sentence) -> Sentence:
        """
        Generate embedding for a single sentence.
        
        Args:
            sentence: Sentence to embed (uses contextualized_text if available)
            
        Returns:
            Sentence with embedding populated
        """
        text = sentence.contextualized_text if sentence.contextualized_text else sentence.text
        sentence.embedding = self.generate_embedding(text)
        return sentence
    
    def embed_sentences(self, sentences: List[Sentence]) -> List[Sentence]:
        """
        Generate embeddings for multiple sentences.
        
        Args:
            sentences: List of sentences to embed
            
        Returns:
            List of sentences with embeddings populated
        """
        return self.openai_service.get_embeddings_for_sentences(sentences)
    
    def validate_embeddings(self, sentences: List[Sentence], 
                          expected_dimensions: int = 1536) -> bool:
        """
        Validate that all sentences have properly dimensioned embeddings.
        
        Args:
            sentences: List of sentences to validate
            expected_dimensions: Expected embedding dimensions
            
        Returns:
            True if all embeddings are valid
        """
        for sentence in sentences:
            if not sentence.embedding:
                raise ValueError(f"Sentence '{sentence.id}' has no embedding")
            
            if len(sentence.embedding) != expected_dimensions:
                raise ValueError(
                    f"Sentence '{sentence.id}' has embedding with {len(sentence.embedding)} "
                    f"dimensions, expected {expected_dimensions}"
                )
        
        return True
    
    def get_embedding_dimensions(self, sentences: List[Sentence]) -> int:
        """
        Get the dimensions of embeddings from a list of sentences.
        
        Args:
            sentences: List of sentences with embeddings
            
        Returns:
            Number of dimensions
        """
        if not sentences:
            raise ValueError("No sentences provided")
        
        first_sentence = sentences[0]
        if not first_sentence.embedding:
            raise ValueError("First sentence has no embedding")
        
        return len(first_sentence.embedding)
