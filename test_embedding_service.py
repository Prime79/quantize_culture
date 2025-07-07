#!/usr/bin/env python3
"""
Unit tests for the EmbeddingService
"""
import pytest
import sys
import os
from unittest.mock import Mock, patch

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('.'))

from app.core.embedding import EmbeddingService
from app.data.models import Sentence


class TestEmbeddingService:
    """Test cases for EmbeddingService."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_openai_service = Mock()
        self.embedding_service = EmbeddingService(self.mock_openai_service)
        
        # Mock embedding vector
        self.mock_embedding = [0.1] * 1536
        self.mock_openai_service.get_embedding.return_value = self.mock_embedding
        self.mock_openai_service.get_embeddings.return_value = [self.mock_embedding, self.mock_embedding]
    
    def test_generate_embedding_single_text(self):
        """Test generating embedding for single text."""
        text = "Test sentence for embedding"
        result = self.embedding_service.generate_embedding(text)
        
        self.mock_openai_service.get_embedding.assert_called_once_with(text)
        assert result == self.mock_embedding
        assert len(result) == 1536
    
    def test_generate_embeddings_multiple_texts(self):
        """Test generating embeddings for multiple texts."""
        texts = ["Text 1", "Text 2"]
        results = self.embedding_service.generate_embeddings(texts)
        
        self.mock_openai_service.get_embeddings.assert_called_once_with(texts)
        assert len(results) == 2
        assert all(len(emb) == 1536 for emb in results)
    
    def test_embed_sentence_with_text(self):
        """Test embedding a sentence using its text."""
        sentence = Sentence(
            id="test_1",
            text="Test sentence text",
            dl_category="Performance & Results",
            dl_subcategory="Results Over Process",
            dl_archetype="Performance & Results - Results Over Process"
        )
        
        result = self.embedding_service.embed_sentence(sentence)
        
        self.mock_openai_service.get_embedding.assert_called_once_with("Test sentence text")
        assert result.embedding == self.mock_embedding
        assert result.id == "test_1"
    
    def test_embed_sentence_with_contextualized_text(self):
        """Test embedding a sentence using its contextualized text."""
        sentence = Sentence(
            id="test_1",
            text="Test sentence text",
            contextualized_text="Contextualized test sentence text",
            dl_category="Performance & Results",
            dl_subcategory="Results Over Process",
            dl_archetype="Performance & Results - Results Over Process"
        )
        
        result = self.embedding_service.embed_sentence(sentence)
        
        # Should use contextualized_text when available
        self.mock_openai_service.get_embedding.assert_called_once_with("Contextualized test sentence text")
        assert result.embedding == self.mock_embedding
    
    def test_embed_sentences_multiple(self):
        """Test embedding multiple sentences."""
        sentences = [
            Sentence(
                id="test_1",
                text="Test sentence 1",
                dl_category="Performance & Results",
                dl_subcategory="Results Over Process",
                dl_archetype="Performance & Results - Results Over Process"
            ),
            Sentence(
                id="test_2",
                text="Test sentence 2",
                dl_category="Innovation & Change",
                dl_subcategory="Fail Fast, Learn Faster",
                dl_archetype="Innovation & Change - Fail Fast, Learn Faster"
            )
        ]
        
        # Mock the get_embeddings_for_sentences method
        self.mock_openai_service.get_embeddings_for_sentences.return_value = sentences
        for sentence in sentences:
            sentence.embedding = self.mock_embedding
        
        results = self.embedding_service.embed_sentences(sentences)
        
        self.mock_openai_service.get_embeddings_for_sentences.assert_called_once_with(sentences)
        assert len(results) == 2
        assert all(sentence.embedding == self.mock_embedding for sentence in results)
    
    def test_validate_embeddings_success(self):
        """Test successful embedding validation."""
        sentences = [
            Sentence(
                id="test_1",
                text="Test sentence 1",
                embedding=[0.1] * 1536,
                dl_category="Performance & Results",
                dl_subcategory="Results Over Process"
            ),
            Sentence(
                id="test_2",
                text="Test sentence 2",
                embedding=[0.2] * 1536,
                dl_category="Innovation & Change",
                dl_subcategory="Fail Fast, Learn Faster"
            )
        ]
        
        result = self.embedding_service.validate_embeddings(sentences)
        assert result is True
    
    def test_validate_embeddings_missing_embedding(self):
        """Test validation failure for missing embedding."""
        sentences = [
            Sentence(
                id="test_1",
                text="Test sentence 1",
                dl_category="Performance & Results",
                dl_subcategory="Results Over Process"
            )
        ]
        
        with pytest.raises(ValueError, match="has no embedding"):
            self.embedding_service.validate_embeddings(sentences)
    
    def test_validate_embeddings_wrong_dimensions(self):
        """Test validation failure for wrong embedding dimensions."""
        sentences = [
            Sentence(
                id="test_1",
                text="Test sentence 1",
                embedding=[0.1] * 100,  # Wrong dimensions
                dl_category="Performance & Results",
                dl_subcategory="Results Over Process"
            )
        ]
        
        with pytest.raises(ValueError, match="has embedding with 100 dimensions, expected 1536"):
            self.embedding_service.validate_embeddings(sentences)
    
    def test_get_embedding_dimensions(self):
        """Test getting embedding dimensions."""
        sentences = [
            Sentence(
                id="test_1",
                text="Test sentence 1",
                embedding=[0.1] * 1536,
                dl_category="Performance & Results",
                dl_subcategory="Results Over Process"
            )
        ]
        
        dimensions = self.embedding_service.get_embedding_dimensions(sentences)
        assert dimensions == 1536
    
    def test_get_embedding_dimensions_no_sentences(self):
        """Test getting dimensions with no sentences."""
        with pytest.raises(ValueError, match="No sentences provided"):
            self.embedding_service.get_embedding_dimensions([])
    
    def test_get_embedding_dimensions_no_embedding(self):
        """Test getting dimensions with sentence that has no embedding."""
        sentences = [
            Sentence(
                id="test_1",
                text="Test sentence 1",
                dl_category="Performance & Results",
                dl_subcategory="Results Over Process"
            )
        ]
        
        with pytest.raises(ValueError, match="First sentence has no embedding"):
            self.embedding_service.get_embedding_dimensions(sentences)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
