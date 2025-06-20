"""
Unit test step definitions for BDD tests
"""
import pytest
from pytest_bdd import scenarios, given, when, then, parsers
import sys
import os
import numpy as np

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'app'))

from embed_and_store import contextualize_sentence, get_embedding
from clustering_optimizer import ClusteringOptimizer
from qualitative_assessment import QualitativeAssessment

# Load scenarios from feature file
scenarios('../features/unit_tests.feature')

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
    return np.random.rand(10, 1536)

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

# Contextualization tests
@given(parsers.parse('a sentence "{sentence}"'))
def sentence_input(sentence):
    return sentence

@when('I call contextualize_sentence with default settings')
def contextualize_default(sentence_input):
    return contextualize_sentence(sentence_input)

@when(parsers.parse('I call contextualize_sentence with context "{context}"'))
def contextualize_custom(sentence_input, context):
    return contextualize_sentence(sentence_input, context_phrase=context)

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
    return contextualize_sentence("Test sentence for embedding")

@when('I call get_embedding')
def call_get_embedding(contextualized_sentence):
    return get_embedding(contextualized_sentence)

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
    return calculate_quantitative_scores(labels, embeddings)

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
    return assess_clustering_qualitative_measures(clustered_sentences)

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
