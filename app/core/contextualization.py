"""Contextualization logic for enhancing sentence embeddings."""

from typing import List, Optional
from ..data.models import Sentence
from ..utils.config import config

class ContextualizationService:
    """Service for adding domain context to sentences."""
    
    def __init__(self, context_prefix: Optional[str] = None, 
                 enable_contextualization: Optional[bool] = None):
        """
        Initialize contextualization service.
        
        Args:
            context_prefix: Custom context prefix (uses config default if None)
            enable_contextualization: Enable/disable contextualization (uses config default if None)
        """
        self.context_prefix = context_prefix or config.domain_context_prefix
        self.enable_contextualization = (
            enable_contextualization 
            if enable_contextualization is not None 
            else config.enable_contextualization
        )
    
    def contextualize_sentence(self, sentence: Sentence) -> Sentence:
        """
        Add domain context to a single sentence.
        
        Args:
            sentence: Sentence object to contextualize
            
        Returns:
            Sentence object with contextualized_text populated
        """
        if not self.enable_contextualization:
            sentence.contextualized_text = sentence.text
            return sentence
        
        # Avoid double-prefixing if already contextualized
        if sentence.text.startswith(self.context_prefix):
            sentence.contextualized_text = sentence.text
        else:
            sentence.contextualized_text = f"{self.context_prefix} {sentence.text}"
        
        return sentence
    
    def contextualize_sentences(self, sentences: List[Sentence]) -> List[Sentence]:
        """
        Add domain context to multiple sentences.
        
        Args:
            sentences: List of Sentence objects to contextualize
            
        Returns:
            List of Sentence objects with contextualized_text populated
        """
        return [self.contextualize_sentence(sentence) for sentence in sentences]
    
    def contextualize_text(self, text: str) -> str:
        """
        Add domain context to raw text.
        
        Args:
            text: Raw text to contextualize
            
        Returns:
            Contextualized text
        """
        if not self.enable_contextualization:
            return text
        
        # Avoid double-prefixing if already contextualized
        if text.startswith(self.context_prefix):
            return text
        else:
            return f"{self.context_prefix} {text}"
    
    def contextualize_texts(self, texts: List[str]) -> List[str]:
        """
        Add domain context to multiple texts.
        
        Args:
            texts: List of raw texts to contextualize
            
        Returns:
            List of contextualized texts
        """
        return [self.contextualize_text(text) for text in texts]
    
    def set_context_prefix(self, prefix: str) -> None:
        """Update the context prefix."""
        self.context_prefix = prefix
    
    def enable(self) -> None:
        """Enable contextualization."""
        self.enable_contextualization = True
    
    def disable(self) -> None:
        """Disable contextualization."""
        self.enable_contextualization = False
