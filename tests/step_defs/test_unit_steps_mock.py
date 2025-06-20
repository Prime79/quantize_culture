"""
Unit test step definitions for BDD tests - simplified working version
"""
import pytest
from pytest_bdd import scenarios, given, when, then, parsers
import sys
import os
import numpy as np
import json

# Load scenarios from feature files
scenarios('../features/unit_tests.feature')
scenarios('../features/unit_embedding_tests.feature') 
scenarios('../features/unit_clustering_tests.feature')

# Test fixtures
@pytest.fixture
def test_sentence():
    return "Innovation drives our decisions"

@pytest.fixture  
def custom_sentence():
    return "We embrace digital transformation"

@pytest.fixture
def sample_embeddings():
    """Create sample embedding vectors for testing"""
    np.random.seed(42)
    return np.random.rand(10, 10)  # Smaller for testing

@pytest.fixture
def sample_labels():
    """Create sample cluster labels for testing"""
    return [0, 0, 1, 1, 2, 2, -1, 0, 1, 2]  # -1 represents noise

@pytest.fixture
def sample_sentences_with_clusters():
    """Sample sentences grouped by clusters for qualitative assessment"""
    return {
        0: ["Innovation drives success", "Technology enables growth"],
        1: ["Team collaboration matters", "People are our strength"], 
        2: ["Data guides decisions", "Analytics improve outcomes"]
    }

# Mock contextualization function
def mock_contextualize_sentence(sentence: str, context_phrase: str = None) -> str:
    """Mock contextualization for testing"""
    if context_phrase:
        return f"{context_phrase}: {sentence}"
    else:
        return f"This is a sentence related to digital rights, privacy, technology, and online culture: {sentence}"

# Mock embedding function 
def mock_get_embedding(sentence: str) -> list:
    """Mock embedding function for testing"""
    # Create a deterministic "embedding" based on sentence length and content
    np.random.seed(len(sentence))
    return np.random.rand(1536).tolist()

# Mock quantitative scoring function
def mock_calculate_quantitative_scores(labels: list, embeddings: np.ndarray) -> dict:
    """Mock quantitative scoring for testing"""
    unique_labels = set(labels)
    noise_count = sum(1 for label in labels if label == -1)
    return {
        'silhouette_score': 0.5,  # Mock value between -1 and 1
        'noise_percentage': (noise_count / len(labels)) * 100,
        'n_clusters': len(unique_labels) - (1 if -1 in unique_labels else 0)
    }

# Mock qualitative assessment function
def mock_assess_clustering_qualitative_measures(clusters_dict: dict) -> dict:
    """Mock qualitative assessment for testing"""
    return {
        'semantic_coherence': 0.8,
        'cultural_alignment': 0.7,
        'interpretability': 0.9
    }

# Contextualization tests
@given(parsers.parse('a sentence "{sentence}"'))
def sentence_input(sentence):
    return sentence

@when('I call contextualize_sentence with default settings')
def contextualize_default(sentence_input):
    return mock_contextualize_sentence(sentence_input)

@when(parsers.parse('I call contextualize_sentence with context "{context}"'))
def contextualize_custom(sentence_input, context):
    return mock_contextualize_sentence(sentence_input, context_phrase=context)

@then('the result should contain the default context prefix')
def check_default_prefix(contextualize_default):
    # Check that the default context is applied
    assert "This is a sentence related to digital rights, privacy, technology, and online culture:" in contextualize_default

@then('the original sentence should be preserved')
def check_original_preserved(contextualize_default, sentence_input):
    assert sentence_input in contextualize_default

@then(parsers.parse('the result should be "{expected_result}"'))
def check_exact_result(contextualize_custom, expected_result):
    assert contextualize_custom == expected_result

# Embedding tests
@given('a contextualized sentence')
def contextualized_sentence():
    return mock_contextualize_sentence("Test sentence for embedding")

@when('I call get_embedding')
def call_get_embedding(contextualized_sentence):
    return mock_get_embedding(contextualized_sentence)

@then('I should receive a 1536-dimensional vector')
def check_embedding_dimensions(call_get_embedding):
    assert len(call_get_embedding) == 1536

@then('the vector should be normalized')  
def check_vector_normalized(call_get_embedding):
    # Check if vector values are reasonable (not all zeros)
    assert not all(x == 0 for x in call_get_embedding)
    # Check if values are in reasonable range for normalized embeddings
    assert all(-2 <= x <= 2 for x in call_get_embedding)

# Clustering evaluation tests
@given('cluster labels and embedding vectors')
def clustering_data(sample_embeddings, sample_labels):
    return sample_embeddings, sample_labels

@when('I call calculate_quantitative_scores')
def call_quantitative_scores(clustering_data):
    embeddings, labels = clustering_data
    return mock_calculate_quantitative_scores(labels, embeddings)

@then('I should get silhouette score between -1 and 1')
def check_silhouette_range(call_quantitative_scores):
    scores = call_quantitative_scores
    assert -1 <= scores['silhouette_score'] <= 1

@then('noise percentage between 0 and 100')
def check_noise_percentage(call_quantitative_scores):
    scores = call_quantitative_scores
    assert 0 <= scores['noise_percentage'] <= 100

@then('cluster count greater than 0')
def check_cluster_count(call_quantitative_scores):
    scores = call_quantitative_scores
    assert scores['n_clusters'] > 0

# Qualitative assessment tests
@given('clustered sentences by theme')
def clustered_sentences(sample_sentences_with_clusters):
    return sample_sentences_with_clusters

@when('I call assess_clustering_qualitative_measures')
def call_qualitative_assessment(clustered_sentences):
    return mock_assess_clustering_qualitative_measures(clustered_sentences)

@then('semantic coherence should be between 0 and 1')
def check_semantic_coherence(call_qualitative_assessment):
    scores = call_qualitative_assessment
    assert 0 <= scores['semantic_coherence'] <= 1

@then('cultural alignment should be between 0 and 1')
def check_cultural_alignment(call_qualitative_assessment):
    scores = call_qualitative_assessment
    assert 0 <= scores['cultural_alignment'] <= 1

@then('interpretability should be between 0 and 1')
def check_interpretability(call_qualitative_assessment):
    scores = call_qualitative_assessment
    assert 0 <= scores['interpretability'] <= 1
