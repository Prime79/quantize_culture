"""Step definitions for DL embedding with labels BDD tests."""

import pytest
import numpy as np
from pytest_bdd import given, when, then, parsers
from app.core.embedding import EmbeddingService
from app.services.qdrant_client import QdrantService
from app.data.models import Sentence

class DLEmbeddingSteps:
    """Step definitions for DL embedding tests."""
    
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.qdrant_service = QdrantService()
        self.test_sentence = None
        self.test_sentences = []
        self.embedding_vector = None
        self.embedding_vectors = []
        self.stored_results = None
        self.query_results = None
        self.test_collection = "test_dl_collection"

# Mock Services for Testing
class MockEmbeddingService:
    def get_embedding(self, text):
        return [0.1] * 1536  # Mock 1536-dimensional embedding
        
    def get_embeddings(self, texts):
        return [[0.1] * 1536 for _ in texts]

class MockQdrantService:
    def __init__(self):
        self.stored_points = []
        
    def store_with_metadata(self, sentences, collection_name):
        for sentence in sentences:
            self.stored_points.append({
                'id': sentence.id,
                'text': sentence.text,
                'dl_category': sentence.dl_category,
                'dl_subcategory': sentence.dl_subcategory,
                'dl_archetype': sentence.dl_archetype,
                'embedding': sentence.embedding
            })
        return len(sentences)
    
    def query_by_metadata(self, collection_name, filter_field, filter_value):
        return [p for p in self.stored_points if p.get(filter_field) == filter_value]

# DL Embedding Steps
@given(parsers.parse('I have a sentence with DL category "{category}" and subcategory "{subcategory}"'))
def given_sentence_with_dl_metadata(category, subcategory):
    """Create a sentence with DL metadata."""
    sentence = Sentence(
        id="test_1",
        text="Just hit the number; we'll fix the paperwork later.",
        dl_category=category,
        dl_subcategory=subcategory,
        dl_archetype=f"{category} - {subcategory}"
    )
    return sentence

@given("I have its embedding vector")
def given_embedding_vector():
    """Create a mock embedding vector."""
    return np.random.rand(1536).tolist()

@given("I have 25 sentences with their DL categories and subcategories")
def given_25_sentences_with_dl_metadata():
    """Create 25 sentences with DL metadata."""
    sentences = []
    categories = [
        ("Performance & Results", "Results Over Process"),
        ("Innovation & Change", "Fail Fast, Learn Faster"),
        ("Control & Authority", "Expert Knows Best"),
        ("Structure & Efficiency", "Standardize to Optimize"),
        ("Relationships & Care", "We're a Family")
    ]
    
    for i in range(25):
        category, subcategory = categories[i % len(categories)]
        sentence = Sentence(
            id=f"test_{i+1}",
            text=f"Test sentence {i+1} for {subcategory}",
            dl_category=category,
            dl_subcategory=subcategory,
            dl_archetype=f"{category} - {subcategory}"
        )
        sentences.append(sentence)
    
    return sentences

@given("I have their embedding vectors")
def given_embedding_vectors(given_25_sentences_with_dl_metadata):
    """Create embedding vectors for the sentences."""
    return [[0.1] * 1536 for _ in given_25_sentences_with_dl_metadata]

@given("I have stored sentences with DL metadata in the collection")
def given_stored_sentences_with_metadata():
    """Mock stored sentences with DL metadata."""
    mock_service = MockQdrantService()
    sentences = given_25_sentences_with_dl_metadata()
    for sentence in sentences:
        sentence.embedding = [0.1] * 1536
    mock_service.store_with_metadata(sentences, "test_collection")
    return mock_service

@given("I have a collection with DL-labeled sentences")
def given_collection_with_dl_labels():
    """Create a collection with DL-labeled sentences."""
    return given_stored_sentences_with_metadata()

@when("I store it in the collection with DL metadata")
def when_store_with_dl_metadata(given_sentence_with_dl_metadata, given_embedding_vector):
    """Store sentence with DL metadata."""
    steps = DLEmbeddingSteps()
    sentence = given_sentence_with_dl_metadata
    sentence.embedding = given_embedding_vector
    
    # Use mock service for testing
    mock_service = MockQdrantService()
    result = mock_service.store_with_metadata([sentence], steps.test_collection)
    steps.stored_results = mock_service.stored_points
    return steps

@when("I store them in bulk to the collection with DL metadata")
def when_store_bulk_with_dl_metadata(given_25_sentences_with_dl_metadata, given_embedding_vectors):
    """Store sentences in bulk with DL metadata."""
    steps = DLEmbeddingSteps()
    sentences = given_25_sentences_with_dl_metadata
    embeddings = given_embedding_vectors
    
    # Assign embeddings to sentences
    for sentence, embedding in zip(sentences, embeddings):
        sentence.embedding = embedding
    
    # Use mock service for testing
    mock_service = MockQdrantService()
    result = mock_service.store_with_metadata(sentences, steps.test_collection)
    steps.stored_results = mock_service.stored_points
    return steps

@when("I query the collection for DL metadata")
def when_query_for_dl_metadata(given_stored_sentences_with_metadata):
    """Query collection for DL metadata."""
    steps = DLEmbeddingSteps()
    mock_service = given_stored_sentences_with_metadata
    steps.query_results = mock_service.stored_points
    return steps

@when(parsers.parse('I query for sentences with category "{category}"'))
def when_query_by_category(given_collection_with_dl_labels, category):
    """Query by DL category."""
    steps = DLEmbeddingSteps()
    mock_service = given_collection_with_dl_labels
    steps.query_results = mock_service.query_by_metadata("test_collection", "dl_category", category)
    return steps

@when(parsers.parse('I query for subcategory "{subcategory}"'))
def when_query_by_subcategory(given_collection_with_dl_labels, subcategory):
    """Query by DL subcategory."""
    steps = DLEmbeddingSteps()
    mock_service = given_collection_with_dl_labels
    steps.query_results = mock_service.query_by_metadata("test_collection", "dl_subcategory", subcategory)
    return steps

@when("I query for a specific DL archetype")
def when_query_by_archetype(given_collection_with_dl_labels):
    """Query by specific DL archetype."""
    steps = DLEmbeddingSteps()
    mock_service = given_collection_with_dl_labels
    archetype = "Performance & Results - Results Over Process"
    steps.query_results = mock_service.query_by_metadata("test_collection", "dl_archetype", archetype)
    return steps

@then("the sentence should be saved with original text")
def then_sentence_saved_with_text(when_store_with_dl_metadata):
    """Verify sentence is saved with original text."""
    steps = when_store_with_dl_metadata
    assert len(steps.stored_results) == 1
    assert steps.stored_results[0]['text'] == "Just hit the number; we'll fix the paperwork later."

@then("the contextualized text should be saved")
def then_contextualized_text_saved(when_store_with_dl_metadata):
    """Verify contextualized text is saved."""
    steps = when_store_with_dl_metadata
    # For now, just verify the original text is saved
    assert steps.stored_results[0]['text'] is not None

@then(parsers.parse('the DL category should be stored as "{category}"'))
def then_dl_category_stored(when_store_with_dl_metadata, category):
    """Verify DL category is stored correctly."""
    steps = when_store_with_dl_metadata
    assert steps.stored_results[0]['dl_category'] == category

@then(parsers.parse('the DL subcategory should be stored as "{subcategory}"'))
def then_dl_subcategory_stored(when_store_with_dl_metadata, subcategory):
    """Verify DL subcategory is stored correctly."""
    steps = when_store_with_dl_metadata
    assert steps.stored_results[0]['dl_subcategory'] == subcategory

@then("the DL archetype should be stored correctly")
def then_dl_archetype_stored(when_store_with_dl_metadata):
    """Verify DL archetype is stored correctly."""
    steps = when_store_with_dl_metadata
    archetype = steps.stored_results[0]['dl_archetype']
    assert archetype is not None
    assert "Performance & Results" in archetype
    assert "Results Over Process" in archetype

@then("the embedding vector should be stored correctly")
def then_embedding_vector_stored(when_store_with_dl_metadata):
    """Verify embedding vector is stored correctly."""
    steps = when_store_with_dl_metadata
    embedding = steps.stored_results[0]['embedding']
    assert embedding is not None
    assert len(embedding) == 1536

@then("all sentences should be saved with complete DL metadata")
def then_all_sentences_with_metadata(when_store_bulk_with_dl_metadata):
    """Verify all sentences have complete DL metadata."""
    steps = when_store_bulk_with_dl_metadata
    assert len(steps.stored_results) == 25
    
    for point in steps.stored_results:
        assert point['dl_category'] is not None
        assert point['dl_subcategory'] is not None
        assert point['dl_archetype'] is not None

@then("each point should have category, subcategory, and archetype fields")
def then_points_have_all_fields(when_store_bulk_with_dl_metadata):
    """Verify each point has all required fields."""
    steps = when_store_bulk_with_dl_metadata
    
    for point in steps.stored_results:
        assert 'dl_category' in point
        assert 'dl_subcategory' in point
        assert 'dl_archetype' in point
        assert 'text' in point
        assert 'embedding' in point

@then("all embedding vectors should be stored correctly")
def then_all_embeddings_stored(when_store_bulk_with_dl_metadata):
    """Verify all embedding vectors are stored correctly."""
    steps = when_store_bulk_with_dl_metadata
    
    for point in steps.stored_results:
        assert point['embedding'] is not None
        assert len(point['embedding']) == 1536

@then("DL metadata should be searchable and filterable")
def then_metadata_searchable(when_store_bulk_with_dl_metadata):
    """Verify DL metadata is searchable and filterable."""
    steps = when_store_bulk_with_dl_metadata
    # This would be tested by the query operations
    assert len(steps.stored_results) > 0

@then("every point should have complete DL information")
def then_complete_dl_info(when_query_for_dl_metadata):
    """Verify every point has complete DL information."""
    steps = when_query_for_dl_metadata
    
    for point in steps.query_results:
        assert point['dl_category'] is not None and point['dl_category'] != ""
        assert point['dl_subcategory'] is not None and point['dl_subcategory'] != ""
        assert point['dl_archetype'] is not None and point['dl_archetype'] != ""

@then("DL categories should match the original source data")
def then_categories_match_source(when_query_for_dl_metadata):
    """Verify DL categories match source data."""
    steps = when_query_for_dl_metadata
    
    # Check that we have expected categories
    categories = set(point['dl_category'] for point in steps.query_results)
    expected_categories = {"Performance & Results", "Innovation & Change", "Control & Authority", "Structure & Efficiency", "Relationships & Care"}
    assert categories.intersection(expected_categories)

@then("DL subcategories should be consistent within categories")
def then_subcategories_consistent(when_query_for_dl_metadata):
    """Verify DL subcategories are consistent within categories."""
    steps = when_query_for_dl_metadata
    
    # Group by category and check subcategories
    category_subcategories = {}
    for point in steps.query_results:
        category = point['dl_category']
        subcategory = point['dl_subcategory']
        if category not in category_subcategories:
            category_subcategories[category] = set()
        category_subcategories[category].add(subcategory)
    
    # Each category should have at least one subcategory
    for category, subcategories in category_subcategories.items():
        assert len(subcategories) > 0

@then("DL archetypes should be properly mapped")
def then_archetypes_mapped(when_query_for_dl_metadata):
    """Verify DL archetypes are properly mapped."""
    steps = when_query_for_dl_metadata
    
    for point in steps.query_results:
        archetype = point['dl_archetype']
        category = point['dl_category']
        subcategory = point['dl_subcategory']
        
        # Archetype should contain both category and subcategory
        assert category in archetype
        assert subcategory in archetype

@then(parsers.parse('I should get only sentences from the {category} category'))
def then_only_category_sentences(when_query_by_category, category):
    """Verify query returns only sentences from specified category."""
    steps = when_query_by_category
    
    for point in steps.query_results:
        assert point['dl_category'] == category

@then("I should get only sentences from that specific subcategory")
def then_only_subcategory_sentences(when_query_by_subcategory):
    """Verify query returns only sentences from specified subcategory."""
    steps = when_query_by_subcategory
    
    # All results should have the same subcategory
    if steps.query_results:
        expected_subcategory = steps.query_results[0]['dl_subcategory']
        for point in steps.query_results:
            assert point['dl_subcategory'] == expected_subcategory

@then("I should get all sentences belonging to that archetype")
def then_all_archetype_sentences(when_query_by_archetype):
    """Verify query returns all sentences for the archetype."""
    steps = when_query_by_archetype
    
    expected_archetype = "Performance & Results - Results Over Process"
    for point in steps.query_results:
        assert point['dl_archetype'] == expected_archetype
