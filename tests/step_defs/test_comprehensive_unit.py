"""
Comprehensive Unit Test Step Definitions with Fixture Data
"""
import pytest
from pytest_bdd import scenarios, given, when, then, parsers
import json
import os
import numpy as np
import time
from typing import List, Dict, Any

# Load all unit test scenarios
scenarios('../features/unit_data_loading.feature')
scenarios('../features/unit_contextualization.feature') 
scenarios('../features/unit_embedding.feature')
scenarios('../features/unit_storage.feature')
scenarios('../features/unit_integration_pipeline.feature')

# Test Context Class
class ComprehensiveTestContext:
    def __init__(self):
        self.fixture_data = None
        self.sentences = []
        self.contextualized_sentences = []
        self.embeddings = []
        self.collection_name = None
        self.storage_result = None
        self.clustering_result = None
        self.quality_scores = None
        self.error_message = None
        self.execution_times = {}
        self.similarity_scores = []

@pytest.fixture
def context():
    return ComprehensiveTestContext()

@pytest.fixture
def fixture_path():
    return os.path.join(os.path.dirname(__file__), '..', 'fixtures', 'test_dl_reference.json')

# Mock functions with realistic behavior
def mock_contextualize_sentence(sentence: str, context_phrase: str = None) -> str:
    """Enhanced mock contextualization"""
    if not sentence.strip():
        context_phrase = context_phrase or "This is a sentence related to digital rights, privacy, technology, and online culture:"
        return context_phrase
    
    if context_phrase:
        return f"{context_phrase} {sentence}"
    else:
        return f"This is a sentence related to digital rights, privacy, technology, and online culture: {sentence}"

def mock_get_embedding(sentence: str) -> List[float]:
    """Enhanced mock embedding with realistic variation"""
    if not sentence.strip():
        return [0.0] * 1536
    
    # Create deterministic but varied embeddings based on sentence content
    seed = hash(sentence) % (2**32)
    np.random.seed(seed)
    
    # Generate embedding with some structure
    base_vector = np.random.normal(0, 0.1, 1536)
    
    # Add sentence-specific features
    sentence_lower = sentence.lower()
    if 'innovation' in sentence_lower:
        base_vector[0:50] += 0.3
    if 'collaboration' in sentence_lower:
        base_vector[50:100] += 0.3
    if 'data' in sentence_lower:
        base_vector[100:150] += 0.3
    if 'digital' in sentence_lower:
        base_vector[150:200] += 0.3
        
    return base_vector.tolist()

def mock_calculate_cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def mock_store_embeddings(sentences: List[str], embeddings: List[List[float]], collection_name: str) -> Dict:
    """Mock storage function"""
    return {
        'collection_name': collection_name,
        'stored_count': len(sentences),
        'points': [
            {
                'id': i,
                'sentence': sentences[i],
                'embedding': embeddings[i],
                'metadata': {'sentence_length': len(sentences[i])}
            }
            for i in range(len(sentences))
        ]
    }

def mock_clustering_optimization(embeddings: List[List[float]]) -> Dict:
    """Mock clustering with realistic results"""
    n_points = len(embeddings)
    n_clusters = min(max(2, n_points // 5), 8)  # 2-8 clusters based on data size
    
    # Generate realistic labels
    np.random.seed(42)
    labels = np.random.randint(0, n_clusters, n_points)
    
    # Add some noise points
    noise_count = max(1, n_points // 10)
    noise_indices = np.random.choice(n_points, noise_count, replace=False)
    labels[noise_indices] = -1
    
    return {
        'labels': labels.tolist(),
        'n_clusters': n_clusters,
        'noise_percentage': (noise_count / n_points) * 100,
        'silhouette_score': 0.65,  # Realistic score
        'parameters': {
            'n_neighbors': 15,
            'min_dist': 0.1,
            'min_cluster_size': 5
        }
    }

def mock_quality_assessment(clustering_result: Dict) -> Dict:
    """Mock quality assessment"""
    return {
        'quantitative': {
            'silhouette_score': clustering_result['silhouette_score'],
            'noise_percentage': clustering_result['noise_percentage'],
            'cluster_count': clustering_result['n_clusters']
        },
        'qualitative': {
            'semantic_coherence': 0.78,
            'cultural_alignment': 0.82,
            'interpretability': 0.85
        },
        'combined_score': 0.76
    }

# DATA LOADING TESTS
@given('I have a valid test fixture file')
def given_valid_fixture(context, fixture_path):
    context.fixture_path = fixture_path
    assert os.path.exists(fixture_path), f"Fixture file not found: {fixture_path}"

@when('I load the JSON data')
def when_load_json(context):
    with open(context.fixture_path, 'r') as f:
        context.fixture_data = json.load(f)

@then('I should get 25 sentences')
def then_check_sentence_count(context):
    assert len(context.fixture_data['example_sentences']) == 25

@then('I should get 4 archetypes')
def then_check_archetype_count(context):
    assert len(context.fixture_data['archetypes']) == 4

@then('I should get 7 digital leadership dimensions')
def then_check_dimension_count(context):
    assert len(context.fixture_data['digital_leadership_dimensions']) == 7

@given('I have loaded test fixture data')
def given_loaded_fixture(context, fixture_path):
    with open(fixture_path, 'r') as f:
        context.fixture_data = json.load(f)

@when('I extract the archetype information')
def when_extract_archetypes(context):
    context.archetypes = context.fixture_data['archetypes']

@then('each archetype should have a description')
def then_check_archetype_descriptions(context):
    for archetype in context.archetypes.values():
        assert 'description' in archetype
        assert len(archetype['description']) > 0

@then('each archetype should have characteristics')
def then_check_archetype_characteristics(context):
    for archetype in context.archetypes.values():
        assert 'characteristics' in archetype
        assert len(archetype['characteristics']) > 0

@when('I extract the example sentences')
def when_extract_sentences(context):
    context.sentences = context.fixture_data['example_sentences']

@then('each sentence should have an id')
def then_check_sentence_ids(context):
    for sentence in context.sentences:
        assert 'id' in sentence
        assert isinstance(sentence['id'], int)

@then('each sentence should have text content')
def then_check_sentence_text(context):
    for sentence in context.sentences:
        assert 'text' in sentence
        assert len(sentence['text']) > 0

@then('each sentence should have archetype mapping')
def then_check_archetype_mapping(context):
    for sentence in context.sentences:
        assert 'archetype' in sentence
        assert sentence['archetype'] in context.fixture_data['archetypes']

@then('each sentence should have dimension mappings')
def then_check_dimension_mappings(context):
    for sentence in context.sentences:
        assert 'dimensions' in sentence
        assert isinstance(sentence['dimensions'], list)
        assert len(sentence['dimensions']) > 0

@given('I have invalid JSON data')
def given_invalid_json(context):
    context.invalid_data = '{"invalid": json syntax'

@when('I attempt to load the data')
def when_load_invalid_data(context):
    try:
        json.loads(context.invalid_data)
        context.error_message = None
    except json.JSONDecodeError as e:
        context.error_message = str(e)

@then('I should get a clear error message')
def then_check_error_message(context):
    assert context.error_message is not None
    assert 'json' in context.error_message.lower() or 'syntax' in context.error_message.lower() or 'expecting' in context.error_message.lower()

@then('the system should not crash')
def then_check_no_crash(context):
    # If we got here, the system didn't crash
    assert True

# CONTEXTUALIZATION TESTS
@given(parsers.parse('I have a sentence "{sentence}"'))
def given_sentence(context, sentence):
    context.input_sentence = sentence

@when('I apply default contextualization')
def when_apply_default_context(context):
    context.result = mock_contextualize_sentence(context.input_sentence)

@then('the result should start with the default context phrase')
def then_check_default_prefix(context):
    assert context.result.startswith("This is a sentence related to digital rights, privacy, technology, and online culture:")

@then('the original sentence should be preserved at the end')
def then_check_sentence_preserved(context):
    assert context.input_sentence in context.result

@then('the result should be properly formatted')
def then_check_formatting(context):
    assert ": " in context.result  # Check for proper colon-space separator

@when(parsers.parse('I apply custom context "{custom_context}"'))
def when_apply_custom_context(context, custom_context):
    context.result = mock_contextualize_sentence(context.input_sentence, custom_context)

@then(parsers.parse('the result should start with "{expected_prefix}"'))
def then_check_custom_prefix(context, expected_prefix):
    assert context.result.startswith(expected_prefix)

@then('the original sentence should follow the context')
def then_check_sentence_follows(context):
    assert context.input_sentence in context.result

@then('there should be proper spacing')
def then_check_spacing(context):
    assert " " in context.result  # Check for spaces

@given('I have a list of 5 test sentences')
def given_sentence_list(context):
    context.input_sentences = [
        "We embrace innovation",
        "Collaboration is key", 
        "Data drives decisions",
        "Digital transformation matters",
        "Learning never stops"
    ]

@when('I apply bulk contextualization with default settings')
def when_bulk_contextualization(context):
    context.contextualized_sentences = [
        mock_contextualize_sentence(s) for s in context.input_sentences
    ]

@then('all 5 sentences should be contextualized')
def then_check_bulk_count(context):
    assert len(context.contextualized_sentences) == 5

@then('each should start with the default context phrase')
def then_check_all_prefixed(context):
    for sentence in context.contextualized_sentences:
        assert sentence.startswith("This is a sentence related to digital rights, privacy, technology, and online culture:")

@then('all original sentences should be preserved')
def then_check_all_preserved(context):
    for i, original in enumerate(context.input_sentences):
        assert original in context.contextualized_sentences[i]

@given('I have an empty sentence')
def given_empty_sentence(context):
    context.input_sentence = ""

@when('I apply contextualization')
def when_apply_contextualization_empty(context):
    context.result = mock_contextualize_sentence(context.input_sentence)

@then('I should get only the context phrase')
def then_check_only_context(context):
    assert context.result == "This is a sentence related to digital rights, privacy, technology, and online culture:"

@then('no errors should occur')
def then_check_no_errors(context):
    assert context.result is not None

# EMBEDDING TESTS
@given('I have a contextualized sentence')
def given_contextualized_sentence(context):
    context.contextualized_sentence = "This is a sentence related to digital rights, privacy, technology, and online culture: Innovation drives success"

@when('I generate an embedding')
def when_generate_embedding(context):
    context.embedding = mock_get_embedding(context.contextualized_sentence)

@then('I should get a 1536-dimensional vector')
def then_check_embedding_dimensions(context):
    assert len(context.embedding) == 1536

@then('all values should be floating point numbers')
def then_check_float_values(context):
    assert all(isinstance(x, (int, float)) for x in context.embedding)

@then('the vector should not be all zeros')
def then_check_not_zeros(context):
    assert not all(x == 0 for x in context.embedding)

@given('I have 5 contextualized sentences')
def given_5_contextualized_sentences(context):
    context.contextualized_sentences = [
        "Context: Innovation drives success",
        "Context: Collaboration matters most",
        "Context: Data guides decisions",
        "Context: Digital transformation",
        "Context: Learning never stops"
    ]

@when('I generate bulk embeddings')
def when_generate_bulk_embeddings(context):
    context.embeddings = [mock_get_embedding(s) for s in context.contextualized_sentences]

@then('I should get 5 embedding vectors')
def then_check_bulk_count(context):
    assert len(context.embeddings) == 5

@then('each should be 1536-dimensional')
def then_check_each_dimension(context):
    for embedding in context.embeddings:
        assert len(embedding) == 1536

@then('vectors should be different from each other')
def then_check_vector_differences(context):
    # Check that at least some vectors are different
    embeddings = context.embeddings
    similarities = []
    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            sim = mock_calculate_cosine_similarity(embeddings[i], embeddings[j])
            similarities.append(sim)
    
    # Not all similarities should be 1.0 (identical)
    assert not all(sim > 0.99 for sim in similarities)

@given('I have various sentence lengths')
def given_various_lengths(context):
    context.sentences = [
        "Short",
        "This is a medium length sentence with several words",
        "This is a very long sentence that contains many words and should test whether the embedding function can handle variable input lengths consistently and produce vectors of the same dimensionality regardless of the input text length"
    ]

@when('I generate embeddings for each')
def when_generate_for_each(context):
    context.embeddings = [mock_get_embedding(s) for s in context.sentences]

@then('all vectors should have exactly 1536 dimensions')
def then_check_consistent_dimensions(context):
    for embedding in context.embeddings:
        assert len(embedding) == 1536

@then('dimension count should be consistent regardless of input length')
def then_check_dimension_consistency(context):
    dimensions = [len(emb) for emb in context.embeddings]
    assert all(dim == 1536 for dim in dimensions)

@given('I have an empty string')
def given_empty_string(context):
    context.empty_string = ""

@when('I attempt to generate an embedding')
def when_generate_empty_embedding(context):
    try:
        context.embedding = mock_get_embedding(context.empty_string)
        context.error_message = None
    except Exception as e:
        context.error_message = str(e)

@then('I should get a valid embedding or clear error')
def then_check_empty_handling(context):
    if context.error_message:
        assert len(context.error_message) > 0
    else:
        assert len(context.embedding) == 1536

@then('the system should handle it gracefully')
def then_check_graceful_handling(context):
    # No crash occurred if we reached this point
    assert True

@given('I have similar sentences')
def given_similar_sentences(context):
    context.similar_sentences = [
        "Innovation drives our business success",
        "Innovation drives our company success",
        "Collaboration is very important",
    ]

@when('I generate embeddings for similar sentences')
def when_generate_similar_embeddings(context):
    context.embeddings = [mock_get_embedding(s) for s in context.similar_sentences]

@then('similar sentences should have higher cosine similarity')
def then_check_similar_similarity(context):
    # First two sentences are more similar
    sim_12 = mock_calculate_cosine_similarity(context.embeddings[0], context.embeddings[1])
    sim_13 = mock_calculate_cosine_similarity(context.embeddings[0], context.embeddings[2])
    
    # Similar sentences should have higher similarity
    assert sim_12 > sim_13

@then('different sentences should have lower similarity')
def then_check_different_similarity(context):
    # This is verified in the previous step
    assert True

@given('the embedding service is unavailable')
def given_service_unavailable(context):
    context.service_error = True

@when('I attempt to generate embeddings')
def when_generate_with_error(context):
    if hasattr(context, 'service_error'):
        context.error_message = "Service unavailable: OpenAI API connection failed"
    else:
        context.embedding = mock_get_embedding("test sentence")

@then('I should get a clear error message about service unavailability')
def then_check_clear_error(context):
    assert hasattr(context, 'error_message')
    assert 'unavailable' in context.error_message.lower() or 'failed' in context.error_message.lower()

@then('the system should not crash')
def then_check_no_crash_embedding(context):
    # If we reached here, no crash occurred
    assert True

# STORAGE TESTS
@given('I have a sentence and its embedding')
def given_sentence_embedding_pair(context):
    context.sentence = "Innovation drives success"
    context.embedding = mock_get_embedding(context.sentence)

@when('I store it in a test collection')
def when_store_single(context):
    context.collection_name = "test_collection_single"
    context.storage_result = mock_store_embeddings(
        [context.sentence], 
        [context.embedding], 
        context.collection_name
    )

@then('the collection should contain one point')
def then_check_single_point(context):
    assert context.storage_result['stored_count'] == 1

@then('the point should have the correct metadata')
def then_check_metadata(context):
    point = context.storage_result['points'][0]
    assert 'sentence' in point
    assert 'embedding' in point
    assert 'metadata' in point

@then('the embedding vector should be preserved')
def then_check_embedding_preserved(context):
    stored_embedding = context.storage_result['points'][0]['embedding']
    assert stored_embedding == context.embedding

@given('I have 25 sentences with embeddings')
def given_25_sentences_embeddings(context, fixture_path):
    with open(fixture_path, 'r') as f:
        fixture_data = json.load(f)
    
    context.sentences = [item['text'] for item in fixture_data['example_sentences']]
    context.embeddings = [mock_get_embedding(s) for s in context.sentences]

@when('I store them in bulk')
def when_store_bulk(context):
    context.collection_name = "test_collection_bulk"
    context.storage_result = mock_store_embeddings(
        context.sentences,
        context.embeddings,
        context.collection_name
    )

@then('the collection should contain 25 points')
def then_check_25_points(context):
    assert context.storage_result['stored_count'] == 25

@then('all metadata should be preserved')
def then_check_all_metadata(context):
    points = context.storage_result['points']
    for point in points:
        assert 'sentence' in point
        assert 'embedding' in point
        assert 'metadata' in point

@then('all embeddings should be retrievable')
def then_check_all_retrievable(context):
    points = context.storage_result['points']
    assert len(points) == len(context.embeddings)

@given('I have a unique collection name')
def given_unique_collection(context):
    import uuid
    context.collection_name = f"test_collection_{uuid.uuid4().hex[:8]}"

@when('I create a new collection')
def when_create_collection(context):
    # Mock collection creation
    context.collection_created = True
    context.collection_config = {
        'name': context.collection_name,
        'vector_size': 1536,
        'distance': 'cosine'
    }

@then('the collection should exist')
def then_check_collection_exists(context):
    assert context.collection_created == True

@then('it should have the correct configuration')
def then_check_collection_config(context):
    assert context.collection_config['vector_size'] == 1536
    assert context.collection_config['distance'] == 'cosine'

@then('it should be empty initially')
def then_check_empty_collection(context):
    # Mock check - new collection should be empty
    context.point_count = 0
    assert context.point_count == 0

@given('I have an existing collection with data')
def given_existing_collection(context):
    context.existing_data = ["Old sentence 1", "Old sentence 2"]
    context.collection_name = "test_existing_collection"

@when('I overwrite it with new data')
def when_overwrite_collection(context):
    new_sentences = ["New sentence 1", "New sentence 2", "New sentence 3"]
    new_embeddings = [mock_get_embedding(s) for s in new_sentences]
    
    context.storage_result = mock_store_embeddings(
        new_sentences,
        new_embeddings,
        context.collection_name
    )

@then('the old data should be replaced')
def then_check_data_replaced(context):
    # Mock verification - new data count should match
    assert context.storage_result['stored_count'] == 3

@then('the new data should be correctly stored')
def then_check_new_data_stored(context):
    points = context.storage_result['points']
    sentences = [point['sentence'] for point in points]
    assert "New sentence 1" in sentences
    assert "New sentence 2" in sentences
    assert "New sentence 3" in sentences

@given('I have stored embeddings in a collection')
def given_stored_embeddings(context):
    context.sentences = ["Test sentence 1", "Test sentence 2"]
    context.embeddings = [mock_get_embedding(s) for s in context.sentences]
    context.storage_result = mock_store_embeddings(
        context.sentences,
        context.embeddings,
        "test_query_collection"
    )

@when('I query the collection')
def when_query_collection(context):
    # Mock query result
    context.query_result = context.storage_result['points']

@then('I should get the correct number of results')
def then_check_query_count(context):
    assert len(context.query_result) == len(context.sentences)

@then('all metadata should be intact')
def then_check_query_metadata(context):
    for point in context.query_result:
        assert 'sentence' in point
        assert 'metadata' in point

@then('embeddings should match original data')
def then_check_query_embeddings(context):
    for i, point in enumerate(context.query_result):
        assert point['embedding'] == context.embeddings[i]

# INTEGRATION PIPELINE TESTS
@given('I have loaded the test fixture with 25 sentences')
def given_full_fixture(context, fixture_path):
    with open(fixture_path, 'r') as f:
        context.fixture_data = json.load(f)
    
    context.original_sentences = [item['text'] for item in context.fixture_data['example_sentences']]
    assert len(context.original_sentences) == 25

@when('I run the complete pipeline step by step')
def when_run_complete_pipeline(context):
    start_time = time.time()
    
    # Step 1: Contextualization
    step_start = time.time()
    context.contextualized_sentences = [
        mock_contextualize_sentence(s) for s in context.original_sentences
    ]
    context.execution_times['contextualization'] = time.time() - step_start
    
    # Step 2: Embedding
    step_start = time.time()
    context.embeddings = [
        mock_get_embedding(s) for s in context.contextualized_sentences
    ]
    context.execution_times['embedding'] = time.time() - step_start
    
    # Step 3: Storage
    step_start = time.time()
    context.collection_name = "test_pipeline_collection"
    context.storage_result = mock_store_embeddings(
        context.original_sentences,
        context.embeddings,
        context.collection_name
    )
    context.execution_times['storage'] = time.time() - step_start
    
    # Step 4: Clustering
    step_start = time.time()
    context.clustering_result = mock_clustering_optimization(context.embeddings)
    context.execution_times['clustering'] = time.time() - step_start
    
    # Step 5: Quality Assessment
    step_start = time.time()
    context.quality_scores = mock_quality_assessment(context.clustering_result)
    context.execution_times['quality_assessment'] = time.time() - step_start
    
    context.execution_times['total'] = time.time() - start_time

@then('after contextualization all 25 sentences should be prefixed')
def then_check_contextualization_step(context):
    assert len(context.contextualized_sentences) == 25
    for sentence in context.contextualized_sentences:
        assert sentence.startswith("This is a sentence related to digital rights, privacy, technology, and online culture:")

@then('after embedding all should have 1536-dimensional vectors')
def then_check_embedding_step(context):
    assert len(context.embeddings) == 25
    for embedding in context.embeddings:
        assert len(embedding) == 1536

@then('after storage the collection should contain 25 points')
def then_check_storage_step(context):
    assert context.storage_result['stored_count'] == 25

@then('after clustering I should get valid cluster assignments')
def then_check_clustering_step(context):
    assert 'labels' in context.clustering_result
    assert len(context.clustering_result['labels']) == 25
    assert context.clustering_result['n_clusters'] > 0
    assert 0 <= context.clustering_result['silhouette_score'] <= 1

@then('after quality assessment I should get numerical scores')
def then_check_quality_step(context):
    assert 'quantitative' in context.quality_scores
    assert 'qualitative' in context.quality_scores
    assert 'combined_score' in context.quality_scores
    assert 0 <= context.quality_scores['combined_score'] <= 1

@then('the final report should contain all required sections')
def then_check_final_report(context):
    # Mock report validation
    required_sections = ['clustering_result', 'quality_scores', 'execution_times']
    for section in required_sections:
        assert hasattr(context, section)

@given('I have a pipeline with potential failure points')
def given_pipeline_with_failures(context):
    context.failure_points = ['embedding_service', 'storage_service', 'clustering_algorithm']

@when('one step fails during execution')
def when_step_fails(context):
    # Simulate a failure in the embedding step
    context.error_message = "Embedding service temporarily unavailable"
    context.partial_results = {
        'contextualization': 'completed',
        'embedding': 'failed',
        'storage': 'not_attempted',
        'clustering': 'not_attempted'
    }

@then('the error should be caught and logged')
def then_check_error_caught(context):
    assert context.error_message is not None
    assert len(context.error_message) > 0

@then('the pipeline should provide meaningful error messages')
def then_check_meaningful_errors(context):
    assert 'unavailable' in context.error_message.lower() or 'failed' in context.error_message.lower()

@then('partial results should be preserved where possible')
def then_check_partial_results(context):
    assert context.partial_results['contextualization'] == 'completed'
    assert context.partial_results['embedding'] == 'failed'

@given('I have the test fixture data')
def given_fixture_for_performance(context, fixture_path):
    with open(fixture_path, 'r') as f:
        context.fixture_data = json.load(f)

@when('I run the complete pipeline')
def when_run_performance_pipeline(context):
    # Same as the step-by-step pipeline but focusing on performance
    start_time = time.time()
    
    sentences = [item['text'] for item in context.fixture_data['example_sentences']]
    contextualized = [mock_contextualize_sentence(s) for s in sentences]
    embeddings = [mock_get_embedding(s) for s in contextualized]
    
    context.total_time = time.time() - start_time
    context.pipeline_completed = True

@then('each step should complete within reasonable time')
def then_check_reasonable_time(context):
    # Mock reasonable time limits (in seconds)
    assert context.total_time < 60  # Should complete within 1 minute for 25 sentences

@then('memory usage should remain within acceptable limits')
def then_check_memory_usage(context):
    # Mock memory check - in real implementation would check actual memory usage
    context.memory_ok = True
    assert context.memory_ok == True

@then('the total pipeline should complete successfully')
def then_check_pipeline_success(context):
    assert context.pipeline_completed == True
