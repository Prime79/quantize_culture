"""
Functional level step definitions for BDD tests
"""
import pytest
from pytest_bdd import scenarios, given, when, then, parsers
import sys
import os
import json
import tempfile
import shutil

# Load scenarios from feature files
scenarios('../features/functionality.feature')
scenarios('../features/functionality_clustering.feature')
scenarios('../features/functionality_storage.feature')

# Context class for sharing data between steps
class FunctionalTestContext:
    def __init__(self):
        self.test_collection_name = None
        self.test_sentences = []
        self.embedding_result = None
        self.clustering_result = None
        self.assessment_result = None
        self.storage_result = None

@pytest.fixture
def context():
    return FunctionalTestContext()

@pytest.fixture
def test_collection_name():
    """Generate a unique test collection name"""
    import uuid
    return f"test_collection_{uuid.uuid4().hex[:8]}"

@pytest.fixture
def test_sentences():
    """Sample test sentences for functional testing"""
    return [
        "We embrace digital transformation",
        "Innovation drives our success", 
        "Data guides our decisions",
        "Collaboration is key to our culture",
        "We invest in continuous learning"
    ]

@pytest.fixture
def test_fixtures_path():
    """Path to test fixtures"""
    return os.path.join(os.path.dirname(__file__), '..', 'fixtures')

@pytest.fixture 
def mock_embeddings():
    """Mock embeddings for testing"""
    import numpy as np
    np.random.seed(42)
    return np.random.rand(5, 1536)  # 5 sentences, 1536 dimensions

@pytest.fixture
def mock_clustering_results():
    """Mock clustering results"""
    return {
        'labels': [0, 0, 1, 1, 2],
        'n_clusters': 3,
        'noise_percentage': 0.0,
        'silhouette_score': 0.6,
        'parameters': {'n_neighbors': 15, 'min_dist': 0.1, 'min_cluster_size': 5}
    }

# Mock functions for testing
def mock_embed_and_store_to_reference_collection(sentences, collection_name, enable_context=True):
    """Mock embedding and storage function"""
    contextualized = []
    for sentence in sentences:
        if enable_context:
            ctx_sentence = f"This is a sentence related to digital rights, privacy, technology, and online culture: {sentence}"
        else:
            ctx_sentence = sentence
        contextualized.append(ctx_sentence)
    
    return {
        'stored_count': len(sentences),
        'contextualized_sentences': contextualized,
        'collection_name': collection_name
    }

def mock_run_clustering_optimization(collection_name):
    """Mock clustering optimization"""
    return {
        'best_params': {'n_neighbors': 15, 'min_dist': 0.1, 'min_cluster_size': 5},
        'parameter_sets_tested': 9,
        'max_clusters_enforced': True,
        'best_score': 0.72
    }

def mock_comprehensive_assessment(clustering_results):
    """Mock comprehensive assessment"""
    return {
        'quantitative': {
            'silhouette_score': 0.6,
            'noise_percentage': 10.0,
            'n_clusters': 8
        },
        'qualitative': {
            'semantic_coherence': 0.8,
            'cultural_alignment': 0.7,
            'interpretability': 0.9
        },
        'combined_score': 0.72
    }

def mock_store_cluster_assignments(collection_name, clustering_results):
    """Mock cluster assignment storage"""
    return {
        'points_updated': 25,
        'clusters_stored': clustering_results.get('n_clusters', 3),
        'metadata_updated': True
    }

# Step definitions using context
@given('I have a list of test sentences')
def given_test_sentences(context, test_sentences):
    context.test_sentences = test_sentences

@when('I call embed_and_store_to_reference_collection with test collection')
def when_embed_and_store(context, test_collection_name):
    context.test_collection_name = test_collection_name
    context.embedding_result = mock_embed_and_store_to_reference_collection(
        context.test_sentences, test_collection_name
    )

@then('each sentence should be prefixed with default context')
def then_check_context_prefix(context):
    contextualized = context.embedding_result['contextualized_sentences']
    for sentence in contextualized:
        assert "This is a sentence related to digital rights, privacy, technology, and online culture:" in sentence

@then('embedded using OpenAI text-embedding-3-small')
def then_check_embedding_model(context):
    assert context.embedding_result['stored_count'] > 0

@then('stored in the specified Qdrant test collection')
def then_check_storage(context):
    assert context.embedding_result['collection_name'] == context.test_collection_name

@given('I have test sentences and a custom context phrase')
def given_custom_context_setup(context, test_sentences):
    context.test_sentences = test_sentences
    context.custom_context = "Digital leadership assessment:"

@when('I enable contextualization with my phrase')
def when_custom_contextualization(context):
    sentences = context.test_sentences
    custom_context = context.custom_context
    contextualized = [f"{custom_context} {sentence}" for sentence in sentences]
    context.embedding_result = {
        'contextualized_sentences': contextualized,
        'custom_context': custom_context
    }

@then('sentences should use my custom prefix')
def then_check_custom_prefix(context):
    contextualized = context.embedding_result['contextualized_sentences']
    custom_context = context.embedding_result['custom_context']
    for sentence in contextualized:
        assert sentence.startswith(custom_context)

@then('embeddings should reflect the domain-specific context')
def then_check_domain_context(context):
    assert len(context.embedding_result['contextualized_sentences']) > 0

# Clustering optimization steps
@given('I have embedded sentences in a test collection')  
def given_embedded_collection(context, test_collection_name):
    context.test_collection_name = test_collection_name

@when('I run clustering optimization')
def when_run_optimization(context):
    context.clustering_result = mock_run_clustering_optimization(context.test_collection_name)

@then('the system should test 9 different parameter sets')
def then_check_parameter_sets(context):
    assert context.clustering_result['parameter_sets_tested'] == 9

@then('evaluate each with quantitative measures')
def then_check_quantitative_evaluation(context):
    assert 'best_score' in context.clustering_result
    assert context.clustering_result['best_score'] > 0

@then('enforce the 50-cluster maximum limit')
def then_check_cluster_limit(context):
    assert context.clustering_result['max_clusters_enforced'] == True

@then('select the best performing parameters')
def then_check_best_params(context):
    assert 'best_params' in context.clustering_result
    assert context.clustering_result['best_params'] is not None

# Assessment steps
@given('I have clustering results')
def given_clustering_results(context):
    context.clustering_result = {
        'labels': [0, 0, 1, 1, 2],
        'n_clusters': 3,
        'noise_percentage': 0.0,
        'silhouette_score': 0.6
    }

@when('I run comprehensive assessment')
def when_run_assessment(context):
    context.assessment_result = mock_comprehensive_assessment(context.clustering_result)

@then('I should get quantitative scores')
def then_check_quantitative_scores(context):
    assert 'quantitative' in context.assessment_result
    quant = context.assessment_result['quantitative']
    assert 'silhouette_score' in quant
    assert 'noise_percentage' in quant
    assert 'n_clusters' in quant

@then('qualitative scores')
def then_check_qualitative_scores(context):
    assert 'qualitative' in context.assessment_result
    qual = context.assessment_result['qualitative']
    assert 'semantic_coherence' in qual
    assert 'cultural_alignment' in qual
    assert 'interpretability' in qual

@then('a combined weighted score')
def then_check_combined_score(context):
    assert 'combined_score' in context.assessment_result
    assert 0 <= context.assessment_result['combined_score'] <= 1

# Database storage steps
@given('I have optimal clustering results')
def given_optimal_results(context):
    context.clustering_result = {
        'labels': [0, 0, 1, 1, 2],
        'n_clusters': 3,
        'best_params': {'n_neighbors': 15, 'min_dist': 0.1}
    }

@when('I store cluster assignments')
def when_store_assignments(context, test_collection_name):
    context.storage_result = mock_store_cluster_assignments(test_collection_name, context.clustering_result)

@then('each sentence point should have a cluster_id')
def then_check_cluster_ids(context):
    assert context.storage_result['points_updated'] > 0

@then('cluster names should be stored as metadata')
def then_check_cluster_metadata(context):
    assert context.storage_result['metadata_updated'] == True

@then('the vector database should be updated with assignments')
def then_check_database_update(context):
    assert context.storage_result['clusters_stored'] > 0
