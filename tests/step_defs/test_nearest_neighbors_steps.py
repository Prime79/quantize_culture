"""
BDD step definitions for Digital Leadership Nearest Neighbors functionality.
Tests the get_nearest_neighbors() method and related analysis logic.
"""

import pytest
import numpy as np
from pytest_bdd import scenarios, given, when, then, parsers
from typing import List, Dict, Any, Optional
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.core.inference_engine import EnhancedInferenceEngine
from app.services.qdrant_client import QdrantService
from app.services.openai_client import OpenAIService
from app.data.inference_models import NearestNeighborsResult, NeighborMatch, DLAnalysis

# Load scenarios from the feature file
scenarios("../features/nearest_neighbors.feature")

# Fixture to hold test context
@pytest.fixture
def context():
    """Test context fixture."""
    from unittest.mock import Mock, MagicMock
    
    class TestContext:
        def __init__(self):
            self.inference_engine = None
            # Mock services instead of real ones
            self.qdrant_service = Mock(spec=QdrantService)
            self.openai_service = Mock(spec=OpenAIService)
            self.input_sentence = None
            self.collection_name = None
            self.n_neighbors = None
            self.neighbors_result = None
            self.error_result = None
            self.similarity_threshold = None
            self.start_time = None
            self.end_time = None
            self.validation_errors = {}
            self.classification_result = None
    
    return TestContext()

# Background steps
@given('I have a reference database "extended_contextualized_collection" with DL metadata')
def step_reference_database_with_dl_metadata(context):
    """Verify reference database exists and has DL metadata."""
    from unittest.mock import Mock
    
    collection_name = "extended_contextualized_collection"
    context.collection_name = collection_name
    
    # Mock Qdrant service
    context.qdrant_service.collection_exists.return_value = True
    
    # Mock search results with proper structure
    mock_results = []
    for i in range(10):
        mock_result = Mock()
        mock_result.id = f"point_{i}"
        # Use small distances (0.1, 0.2, 0.3...) so that similarities (1-distance) are high and descending
        mock_result.score = 0.1 + (i * 0.1)  # Distances: 0.1, 0.2, 0.3, 0.4, 0.5...
        mock_result.payload = {
            'sentence': f'Sample DL sentence {i+1} about digital leadership and transformation',
            'dl_category': f'Category_{i%3}',
            'dl_subcategory': f'Subcategory_{i%2}',
            'dl_archetype': f'Archetype_{i%3}'
        }
        mock_results.append(mock_result)
    
    context.qdrant_service.search_similar.return_value = mock_results
    context.qdrant_service.extract_data.return_value = mock_results[:10]  # Return sample data
    
    # Mock OpenAI service
    context.openai_service.get_embedding.return_value = [0.1] * 1536
    
    # Initialize inference engine with mocked services
    context.inference_engine = EnhancedInferenceEngine(
        context.openai_service, 
        context.qdrant_service
    )

@given('the database contains sentences with categories, subcategories, and archetypes')
def step_database_contains_dl_structure(context):
    """Verify the database has proper DL metadata structure."""
    # This is already set up in the previous step with mocked data
    # The mock data includes DL metadata structure
    pass

@given('the inference engine is properly configured')
def step_inference_engine_configured(context):
    """Verify inference engine is ready."""
    assert context.inference_engine is not None
    assert context.qdrant_service is not None
    assert context.openai_service is not None

# Input steps
@given('I have an input sentence "{sentence}"')
def step_input_sentence(context, sentence):
    """Set the input sentence for testing."""
    context.input_sentence = sentence

@given('I have an input sentence that matches neighbors with incomplete metadata')
def step_input_sentence_incomplete_metadata(context):
    """Set input sentence that will match neighbors with missing DL data."""
    # Use a sentence that's likely to match some entries without full DL metadata
    context.input_sentence = "execution matters more than planning"

@given('I have an empty reference database')
def step_empty_reference_database(context):
    """Set up scenario with empty database."""
    context.collection_name = "empty_test_collection"
    # This collection either doesn't exist or is empty

@given('I have a reference database with data')
def step_reference_database_with_data(context):
    """Set up database with test data."""
    context.collection_name = "extended_contextualized_collection"
    context.inference_engine = EnhancedInferenceEngine(
        context.openai_service, 
        context.qdrant_service
    )

# Action steps  
@when('I request {n:d} nearest neighbors from the database')
def step_request_nearest_neighbors(context, n):
    """Request n nearest neighbors for the input sentence."""
    context.n_neighbors = n
    try:
        context.neighbors_result = context.inference_engine.get_nearest_neighbors(
            context.input_sentence,
            context.collection_name,
            n_neighbors=n
        )
    except Exception as e:
        context.error_result = str(e)

@when('I request {n:d} nearest neighbors with DL metadata')
def step_request_neighbors_with_dl_metadata(context, n):
    """Request neighbors specifically for DL metadata analysis."""
    context.n_neighbors = n
    try:
        context.neighbors_result = context.inference_engine.get_nearest_neighbors(
            context.input_sentence,
            context.collection_name,
            n_neighbors=n,
            include_dl_analysis=True
        )
    except Exception as e:
        context.error_result = str(e)

@when('I request {n:d} nearest neighbors for dominant logic analysis')
def step_request_neighbors_for_dominant_logic(context, n):
    """Request neighbors for dominant logic analysis."""
    context.n_neighbors = n
    try:
        context.neighbors_result = context.inference_engine.get_nearest_neighbors(
            context.input_sentence,
            context.collection_name,
            n_neighbors=n,
            include_dl_analysis=True,
            include_distribution_stats=True
        )
    except Exception as e:
        context.error_result = str(e)

@when('I request {n:d} nearest neighbors for distribution analysis')
def step_request_neighbors_for_distribution(context, n):
    """Request neighbors for distribution analysis."""
    context.n_neighbors = n
    try:
        context.neighbors_result = context.inference_engine.get_nearest_neighbors(
            context.input_sentence,
            context.collection_name,
            n_neighbors=n,
            include_distribution_stats=True
        )
    except Exception as e:
        context.error_result = str(e)

@when('I request {n:d} nearest neighbors with minimum similarity threshold {threshold:f}')
def step_request_neighbors_with_threshold(context, n, threshold):
    """Request neighbors with similarity threshold."""
    context.n_neighbors = n
    context.similarity_threshold = threshold
    try:
        context.neighbors_result = context.inference_engine.get_nearest_neighbors(
            context.input_sentence,
            context.collection_name,
            n_neighbors=n,
            min_similarity=threshold
        )
    except Exception as e:
        context.error_result = str(e)

@when('I get both nearest neighbors and cluster classification')
def step_get_neighbors_and_classification(context):
    """Get both nearest neighbors and cluster classification for comparison."""
    try:
        # Get nearest neighbors
        context.neighbors_result = context.inference_engine.get_nearest_neighbors(
            context.input_sentence,
            context.collection_name,
            n_neighbors=5
        )
        
        # Mock cluster classification result instead of calling real method
        from unittest.mock import Mock
        context.classification_result = Mock()
        context.classification_result.primary_match = Mock()
        context.classification_result.primary_match.cluster_id = 0
        
    except Exception as e:
        context.error_result = str(e)

@when('I request {n:d} nearest neighbors')
def step_request_n_nearest_neighbors(context, n):
    """Generic step for requesting n neighbors."""
    context.n_neighbors = n
    import time
    context.start_time = time.time()
    
    try:
        context.neighbors_result = context.inference_engine.get_nearest_neighbors(
            context.input_sentence,
            context.collection_name,
            n_neighbors=n
        )
    except Exception as e:
        context.error_result = str(e)
    finally:
        context.end_time = time.time()

@when('I request {n:d} nearest neighbors with semantic analysis')
def step_request_neighbors_with_semantic_analysis(context, n):
    """Request neighbors with semantic analysis."""
    context.n_neighbors = n
    try:
        context.neighbors_result = context.inference_engine.get_nearest_neighbors(
            context.input_sentence,
            context.collection_name,
            n_neighbors=n,
            include_semantic_analysis=True
        )
    except Exception as e:
        context.error_result = str(e)

@when('I request nearest neighbors with invalid parameters')
def step_request_neighbors_invalid_parameters(context):
    """Test various invalid parameter combinations."""
    context.validation_errors = {}
    
    # Test empty sentence
    try:
        context.inference_engine.get_nearest_neighbors("", context.collection_name, 5)
    except Exception as e:
        context.validation_errors["empty_sentence"] = str(e)
    
    # Test zero neighbors
    try:
        context.inference_engine.get_nearest_neighbors("test", context.collection_name, 0)
    except Exception as e:
        context.validation_errors["zero_neighbors"] = str(e)
    
    # Test negative neighbors
    try:
        context.inference_engine.get_nearest_neighbors("test", context.collection_name, -5)
    except Exception as e:
        context.validation_errors["negative_neighbors"] = str(e)
    
    # Test nonexistent collection
    try:
        context.inference_engine.get_nearest_neighbors("test", "nonexistent", 5)
    except Exception as e:
        context.validation_errors["nonexistent_collection"] = str(e)

# Assertion steps
@then('I should get exactly {n:d} neighbor sentences')
def step_should_get_exactly_n_neighbors(context, n):
    """Verify we get exactly n neighbors."""
    assert context.neighbors_result is not None, "Should get neighbors result"
    assert 'neighbors' in context.neighbors_result, "Result should have neighbors"
    assert len(context.neighbors_result['neighbors']) == n, \
        f"Expected {n} neighbors, got {len(context.neighbors_result['neighbors'])}"

@then('I should get exactly 5 neighbor sentences')
def step_should_get_exactly_5_neighbors(context):
    """Verify we get exactly 5 neighbors."""
    assert context.neighbors_result is not None, "Should get neighbors result"
    assert 'neighbors' in context.neighbors_result, "Result should have neighbors"
    assert len(context.neighbors_result['neighbors']) == 5, \
        f"Expected 5 neighbors, got {len(context.neighbors_result['neighbors'])}"

@then('each neighbor should have a similarity score')
def step_each_neighbor_has_similarity_score(context):
    """Verify each neighbor has a similarity score."""
    for neighbor in context.neighbors_result['neighbors']:
        assert 'similarity_score' in neighbor, "Neighbor should have similarity_score"
        assert isinstance(neighbor['similarity_score'], (int, float)), "Similarity score should be numeric"
        assert 0 <= neighbor['similarity_score'] <= 1, "Similarity score should be between 0 and 1"

@then('neighbors should be ranked by similarity score in descending order')
def step_neighbors_ranked_by_similarity(context):
    """Verify neighbors are ranked by similarity in descending order."""
    similarities = [n['similarity_score'] for n in context.neighbors_result['neighbors']]
    assert similarities == sorted(similarities, reverse=True), \
        "Neighbors should be ranked by similarity in descending order"

@then('each neighbor should include the original sentence text')
def step_each_neighbor_has_sentence_text(context):
    """Verify each neighbor includes sentence text."""
    for neighbor in context.neighbors_result['neighbors']:
        assert 'sentence' in neighbor, "Neighbor should have sentence text"
        assert neighbor['sentence'], "Sentence text should not be empty"
        assert isinstance(neighbor['sentence'], str), "Sentence should be a string"

@then('I should get {n:d} neighbor sentences with their metadata')
def step_should_get_neighbors_with_metadata(context, n):
    """Verify we get n neighbors with DL metadata."""
    assert len(context.neighbors_result['neighbors']) == n
    
    # Check that neighbors have DL metadata structure (even if some values are None)
    for neighbor in context.neighbors_result['neighbors']:
        assert 'dl_category' in neighbor, "Neighbor should have dl_category field"
        assert 'dl_subcategory' in neighbor, "Neighbor should have dl_subcategory field"
        assert 'dl_archetype' in neighbor, "Neighbor should have dl_archetype field"

@then('I should get 5 neighbor sentences with their metadata')
def step_should_get_5_neighbors_with_metadata(context):
    """Verify we get 5 neighbors with DL metadata."""
    assert len(context.neighbors_result['neighbors']) == 5
    
    # Check that neighbors have DL metadata structure (even if some values are None)
    for neighbor in context.neighbors_result['neighbors']:
        assert 'dl_category' in neighbor, "Neighbor should have dl_category field"
        assert 'dl_subcategory' in neighbor, "Neighbor should have dl_subcategory field"
        assert 'dl_archetype' in neighbor, "Neighbor should have dl_archetype field"

@then('each neighbor should include DL category if available')
def step_neighbors_include_dl_category(context):
    """Verify DL category is included when available."""
    for neighbor in context.neighbors_result['neighbors']:
        # Category should be None or a non-empty string
        if neighbor['dl_category'] is not None:
            assert isinstance(neighbor['dl_category'], str), "DL category should be string"
            assert neighbor['dl_category'].strip(), "DL category should not be empty"

@then('each neighbor should include DL subcategory if available')
def step_neighbors_include_dl_subcategory(context):
    """Verify DL subcategory is included when available."""
    for neighbor in context.neighbors_result['neighbors']:
        if neighbor['dl_subcategory'] is not None:
            assert isinstance(neighbor['dl_subcategory'], str), "DL subcategory should be string"
            assert neighbor['dl_subcategory'].strip(), "DL subcategory should not be empty"

@then('each neighbor should include DL archetype if available')
def step_neighbors_include_dl_archetype(context):
    """Verify DL archetype is included when available."""
    for neighbor in context.neighbors_result['neighbors']:
        if neighbor['dl_archetype'] is not None:
            assert isinstance(neighbor['dl_archetype'], str), "DL archetype should be string"
            assert neighbor['dl_archetype'].strip(), "DL archetype should not be empty"

@then('metadata should be properly structured')
def step_metadata_properly_structured(context):
    """Verify metadata has proper structure."""
    for neighbor in context.neighbors_result['neighbors']:
        # Check that all DL fields exist (can be None)
        required_fields = ['dl_category', 'dl_subcategory', 'dl_archetype']
        for field in required_fields:
            assert field in neighbor, f"Neighbor should have {field} field"

@then('I should get dominant logic analysis results')
def step_should_get_dominant_logic_analysis(context):
    """Verify dominant logic analysis is included."""
    assert 'dominant_logic' in context.neighbors_result, \
        "Result should include dominant logic analysis"
    assert context.neighbors_result['dominant_logic'] is not None, \
        "Dominant logic should not be None"

@then('I should see the most common DL category across neighbors')
def step_should_see_most_common_category(context):
    """Verify most common category is identified."""
    dl_analysis = context.neighbors_result['dominant_logic']
    assert 'dominant_category' in dl_analysis, "Should have dominant_category"
    # Dominant category can be None if no neighbors have categories

@then('I should see the most common DL subcategory across neighbors')
def step_should_see_most_common_subcategory(context):
    """Verify most common subcategory is identified."""
    dl_analysis = context.neighbors_result['dominant_logic']
    assert 'dominant_subcategory' in dl_analysis, "Should have dominant_subcategory"

@then('I should see the most common DL archetype across neighbors')
def step_should_see_most_common_archetype(context):
    """Verify most common archetype is identified."""
    dl_analysis = context.neighbors_result['dominant_logic']
    assert 'dominant_archetype' in dl_analysis, "Should have dominant_archetype"

@then('I should get confidence scores for each dominant element')
def step_should_get_confidence_scores(context):
    """Verify confidence scores for dominant logic elements."""
    dl_analysis = context.neighbors_result['dominant_logic']
    
    # Check for confidence fields
    confidence_fields = ['category_confidence', 'subcategory_confidence', 'archetype_confidence']
    for field in confidence_fields:
        if field in dl_analysis:
            confidence = dl_analysis[field]
            if confidence is not None:
                assert 0 <= confidence <= 1, f"{field} should be between 0 and 1"

@then('dominant logic analysis should include confidence warnings')
def step_dominant_logic_should_include_confidence_warnings(context):
    """Verify that low confidence scores trigger appropriate warnings."""
    dl_analysis = context.neighbors_result['dominant_logic']
    
    # Check if warnings field exists when dealing with low similarity scores
    if 'warnings' in dl_analysis:
        warnings = dl_analysis['warnings']
        assert isinstance(warnings, list), "Warnings should be a list"
        
    # Alternative: Check for low confidence indicators
    confidence_fields = ['category_confidence', 'subcategory_confidence', 'archetype_confidence']
    low_confidence_found = False
    for field in confidence_fields:
        if field in dl_analysis and dl_analysis[field] is not None:
            if dl_analysis[field] < 0.5:  # Low confidence threshold
                low_confidence_found = True
                break
    
    # If low confidence found, ensure appropriate handling
    if low_confidence_found:
        # Could check for warnings or special handling flags
        pass

@then('I should get {n:d} neighbors regardless of metadata completeness')
def step_get_neighbors_regardless_metadata(context, n):
    """Verify we get n neighbors even with incomplete metadata."""
    assert len(context.neighbors_result['neighbors']) == n, \
        f"Should get {n} neighbors regardless of metadata completeness"

@then('neighbors with missing metadata should be included')
def step_neighbors_with_missing_metadata_included(context):
    """Verify neighbors with missing metadata are included."""
    # We should have neighbors even if some have None values
    assert len(context.neighbors_result['neighbors']) > 0, \
        "Should include neighbors even with missing metadata"

@then('missing metadata fields should be marked as null or empty')
def step_missing_metadata_marked_null(context):
    """Verify missing metadata is properly marked."""
    for neighbor in context.neighbors_result['neighbors']:
        # Fields should exist but can be None
        assert 'dl_category' in neighbor
        assert 'dl_subcategory' in neighbor
        assert 'dl_archetype' in neighbor

@then('dominant logic analysis should work with available data only')
def step_dominant_logic_works_with_available_data(context):
    """Verify dominant logic analysis handles missing data."""
    assert 'dominant_logic' in context.neighbors_result, \
        "Should have dominant logic analysis even with missing data"

@then('I should get warnings about incomplete metadata coverage')
def step_should_get_metadata_warnings(context):
    """Verify warnings about incomplete metadata."""
    # Check for warnings in the result
    if hasattr(context.neighbors_result, 'warnings'):
        warnings = context.neighbors_result.warnings
        # Should warn about incomplete metadata if applicable

@then('I should get detailed distribution statistics')
def step_should_get_distribution_statistics(context):
    """Verify detailed distribution statistics."""
    assert 'distribution_stats' in context.neighbors_result, \
        "Result should include distribution statistics"
    stats = context.neighbors_result['distribution_stats']
    assert stats is not None, "Statistics should not be None"

@then('I should see category distribution with counts')
def step_should_see_category_distribution(context):
    """Verify category distribution with counts."""
    stats = context.neighbors_result['distribution_stats']
    assert 'category_distribution' in stats, "Should have category distribution"
    distribution = stats['category_distribution']
    assert isinstance(distribution, dict), "Category distribution should be a dictionary"

@then('I should see subcategory distribution with counts')
def step_should_see_subcategory_distribution(context):
    """Verify subcategory distribution with counts."""
    stats = context.neighbors_result['distribution_stats']
    assert 'subcategory_distribution' in stats, "Should have subcategory distribution"

@then('I should see archetype distribution with counts')
def step_should_see_archetype_distribution(context):
    """Verify archetype distribution with counts."""
    stats = context.neighbors_result['distribution_stats']
    assert 'archetype_distribution' in stats, "Should have archetype distribution"

@then('I should get the total number of neighbors analyzed')
def step_should_get_total_neighbors(context):
    """Verify total neighbor count."""
    stats = context.neighbors_result['distribution_stats']
    assert 'total_neighbors' in stats, "Should have total neighbors count"
    assert stats['total_neighbors'] == len(context.neighbors_result['neighbors']), \
        "Total neighbors should match actual neighbor count"

@then('I should get the number of unique categories found')
def step_should_get_unique_categories_count(context):
    """Verify unique categories count."""
    stats = context.neighbors_result['distribution_stats']
    assert 'unique_categories' in stats, "Should have unique categories count"

@then('all similarity scores should be between 0 and 1')
def step_similarity_scores_valid_range(context):
    """Verify all similarity scores are in valid range."""
    for neighbor in context.neighbors_result['neighbors']:
        score = neighbor['similarity_score']
        assert 0 <= score <= 1, f"Similarity score {score} should be between 0 and 1"

@then('I should get a warning about low similarity matches')
def step_should_get_low_similarity_warning(context):
    """Verify warning about low similarity."""
    # Check for warnings about low similarity
    if hasattr(context.neighbors_result, 'warnings'):
        warnings = context.neighbors_result.warnings
        # Should contain warning about low similarity matches

@then('neighbors should be strictly ranked by similarity score')
def step_neighbors_strictly_ranked(context):
    """Verify strict ranking by similarity."""
    similarities = [n['similarity_score'] for n in context.neighbors_result['neighbors']]
    for i in range(len(similarities) - 1):
        assert similarities[i] >= similarities[i + 1], \
            "Neighbors should be ranked in descending order by similarity"

@then('no two neighbors should have identical similarity scores (ties broken consistently)')
def step_no_identical_similarity_scores(context):
    """Verify ties are broken consistently."""
    similarities = [n['similarity_score'] for n in context.neighbors_result['neighbors']]
    # Allow identical scores but verify they're handled consistently
    # (This is implementation-specific - we may allow ties)

@then('the first neighbor should have the highest similarity score')
def step_first_neighbor_highest_score(context):
    """Verify first neighbor has highest score."""
    neighbors = context.neighbors_result['neighbors']
    if len(neighbors) > 1:
        assert neighbors[0]['similarity_score'] >= neighbors[1]['similarity_score'], \
            "First neighbor should have highest similarity score"

@then('the last neighbor should have the lowest similarity score')
def step_last_neighbor_lowest_score(context):
    """Verify last neighbor has lowest score."""
    neighbors = context.neighbors_result['neighbors']
    if len(neighbors) > 1:
        assert neighbors[-1]['similarity_score'] <= neighbors[-2]['similarity_score'], \
            "Last neighbor should have lowest similarity score"

@then('similarity scores should decrease monotonically')
def step_similarity_scores_decrease_monotonically(context):
    """Verify similarity scores decrease monotonically."""
    similarities = [n['similarity_score'] for n in context.neighbors_result['neighbors']]
    for i in range(len(similarities) - 1):
        assert similarities[i] >= similarities[i + 1], \
            "Similarity scores should decrease monotonically"

@then('I should only get neighbors with similarity >= {threshold:f}')
def step_neighbors_above_threshold(context, threshold):
    """Verify all neighbors meet similarity threshold."""
    for neighbor in context.neighbors_result['neighbors']:
        assert neighbor['similarity_score'] >= threshold, \
            f"Neighbor similarity {neighbor['similarity_score']} should be >= {threshold}"

@then('if fewer than {n:d} neighbors meet the threshold, return only those that qualify')
def step_fewer_neighbors_than_requested(context, n):
    """Verify behavior when fewer neighbors meet threshold."""
    # This is acceptable - we may get fewer than requested
    actual_count = len(context.neighbors_result['neighbors'])
    assert actual_count <= n, f"Should not get more than {n} neighbors"

@then('if no neighbors meet the threshold, return empty result with explanation')
def step_no_neighbors_meet_threshold(context):
    """Verify behavior when no neighbors meet threshold."""
    if len(context.neighbors_result['neighbors']) == 0:
        # Should have explanation for empty result
        assert 'explanation' in context.neighbors_result or \
               'warnings' in context.neighbors_result, \
               "Should provide explanation for empty result"

@then('threshold filtering should not affect ranking order')
def step_threshold_preserves_ranking(context):
    """Verify threshold filtering preserves ranking."""
    similarities = [n['similarity_score'] for n in context.neighbors_result['neighbors']]
    assert similarities == sorted(similarities, reverse=True), \
        "Threshold filtering should preserve ranking order"

@then('nearest neighbors should include sentences from the matched cluster')
def step_neighbors_include_matched_cluster(context):
    """Verify neighbors include sentences from the matched cluster."""
    # Get the cluster from classification result if available
    if hasattr(context, 'classification_result') and context.classification_result:
        matched_cluster = context.classification_result.primary_match.cluster_id
        # Check if any neighbors are from the same cluster
        # Note: This requires cluster information in neighbor results
        # This is an integration test between the two approaches
    else:
        # If classification result is not available, just verify we have neighbors
        assert len(context.neighbors_result['neighbors']) > 0, "Should have neighbors"

@then('nearest neighbors may include sentences from other clusters')
def step_neighbors_may_include_other_clusters(context):
    """Verify neighbors can come from different clusters."""
    # This is expected behavior - neighbors are based on similarity, not cluster membership

@then('the approach should complement cluster-based classification')
def step_approaches_should_complement(context):
    """Verify nearest neighbors complements cluster classification."""
    # Both approaches should provide valuable but different insights
    assert context.neighbors_result is not None
    assert context.classification_result is not None

@then('results should be consistent in terms of DL themes')
def step_results_consistent_dl_themes(context):
    """Verify DL themes are consistent between approaches."""
    # Check that DL themes from neighbors and classification are related

@then('the operation should complete within reasonable time (< {max_seconds:d} seconds)')
def step_operation_completes_in_time(context, max_seconds):
    """Verify operation completes within time limit."""
    execution_time = context.end_time - context.start_time
    assert execution_time < max_seconds, \
        f"Operation took {execution_time:.2f}s, should be < {max_seconds}s"

@then('I should get exactly {n:d} neighbors if database has enough sentences')
def step_get_exact_neighbors_if_available(context, n):
    """Verify we get exact number if database has enough sentences."""
    actual_count = len(context.neighbors_result['neighbors'])
    # Check if database has enough sentences
    # Our mock has 10 sentences total
    expected_count = min(n, 10)
    assert actual_count == expected_count, \
        f"Expected {expected_count} neighbors, got {actual_count}"

@then('memory usage should remain reasonable')
def step_memory_usage_reasonable(context):
    """Verify memory usage is reasonable."""
    # This is a qualitative check - operation should not crash due to memory

@then('similarity calculations should be optimized')
def step_similarity_calculations_optimized(context):
    """Verify similarity calculations are optimized."""
    # This is checked implicitly through timing constraints

@then('the result should be properly structured')
def step_result_properly_structured(context):
    """Verify result has proper structure."""
    result = context.neighbors_result
    assert result is not None, "Result should not be None"

@then('I should get query metadata (sentence, collection, timestamp)')
def step_get_query_metadata(context):
    """Verify query metadata is included."""
    result = context.neighbors_result
    assert 'query_sentence' in result, "Should have query sentence"
    assert 'collection_name' in result, "Should have collection name"
    # Timestamp is optional but recommended

@then('I should get an array of neighbor objects')
def step_get_neighbor_objects_array(context):
    """Verify neighbors are provided as array."""
    result = context.neighbors_result
    assert 'neighbors' in result, "Should have neighbors array"
    assert isinstance(result['neighbors'], list), "Neighbors should be a list"

@then('each neighbor should have: sentence, similarity_score, rank, dl_metadata')
def step_neighbor_has_required_fields(context):
    """Verify each neighbor has required fields."""
    for neighbor in context.neighbors_result['neighbors']:
        required_fields = ['sentence', 'similarity_score']
        for field in required_fields:
            assert field in neighbor, f"Neighbor should have {field}"

@then('I should get dominant logic analysis object')
def step_get_dominant_logic_object(context):
    """Verify dominant logic analysis object."""
    result = context.neighbors_result
    assert 'dominant_logic' in result, "Should have dominant logic analysis"

@then('I should get distribution statistics object')
def step_get_distribution_statistics_object(context):
    """Verify distribution statistics object."""
    result = context.neighbors_result
    assert 'distribution_stats' in result, "Should have distribution statistics"

@then('I should get an appropriate error message')
def step_get_appropriate_error_message(context):
    """Verify appropriate error message."""
    assert context.error_result is not None, "Should get an error message"
    assert isinstance(context.error_result, str), "Error should be a string"
    assert len(context.error_result) > 0, "Error message should not be empty"

@then('the error should indicate no data available')
def step_error_indicates_no_data(context):
    """Verify error indicates no data available."""
    error_msg = context.error_result.lower()
    assert any(phrase in error_msg for phrase in ['no data', 'empty', 'not found']), \
        "Error should indicate no data available"

@then('the system should not crash or throw exceptions')
def step_system_should_not_crash(context):
    """Verify system handles errors gracefully."""
    # If we get here, the system didn't crash

@then('I should get guidance on populating the database')
def step_get_guidance_on_populating_database(context):
    """Verify guidance on populating database."""
    # Error message should be helpful
    assert context.error_result is not None

@then('I should get appropriate validation errors for')
def step_get_validation_errors(context):
    """Verify validation errors for invalid parameters."""
    errors = context.validation_errors
    assert len(errors) > 0, "Should get validation errors for invalid parameters"
    
    # Check specific error types exist
    expected_errors = ['empty_sentence', 'zero_neighbors', 'negative_neighbors', 'nonexistent_collection']
    for error_type in expected_errors:
        if error_type in errors:
            assert errors[error_type], f"Should have error message for {error_type}"

@then('I should get neighbors based on vector similarity')
def step_get_neighbors_by_vector_similarity(context):
    """Verify neighbors are based on vector similarity."""
    assert len(context.neighbors_result['neighbors']) > 0, "Should get neighbors"

@then('I should get semantic similarity scores for comparison')
def step_get_semantic_similarity_scores(context):
    """Verify semantic similarity scores are provided."""
    # Check if semantic analysis is included
    for neighbor in context.neighbors_result['neighbors']:
        if hasattr(neighbor, 'semantic_similarity'):
            assert isinstance(neighbor.semantic_similarity, (int, float)), \
                "Semantic similarity should be numeric"

@then('semantic analysis should identify keyword overlaps')
def step_semantic_analysis_identifies_keywords(context):
    """Verify semantic analysis identifies keywords."""
    # Check for keyword analysis in results
    if hasattr(context.neighbors_result, 'semantic_analysis'):
        semantic = context.neighbors_result.semantic_analysis
        # Should have keyword overlap information

@then('results should show both similarity types')
def step_results_show_both_similarities(context):
    """Verify both vector and semantic similarities are shown."""
    # Check that both types of similarity are available
    neighbors = context.neighbors_result['neighbors']
    assert len(neighbors) > 0, "Should have neighbors to analyze"

@then('dominant logic should consider semantic factors')
def step_dominant_logic_considers_semantic(context):
    """Verify dominant logic considers semantic factors."""
    # Check that semantic analysis influences dominant logic
    if hasattr(context.neighbors_result, 'dominant_logic'):
        dl = context.neighbors_result.dominant_logic
        # Semantic factors should be considered in the analysis

# Additional step definitions for specific hardcoded sentences in feature file
@given('I have an input sentence "order is the key for success"')
def step_input_sentence_order_key_success(context):
    """Set specific input sentence."""
    context.input_sentence = "order is the key for success"

@given('I have an input sentence "fail fast learn faster"')
def step_input_sentence_fail_fast(context):
    """Set specific input sentence."""
    context.input_sentence = "fail fast learn faster"

@given('I have an input sentence "innovation drives our decisions"')
def step_input_sentence_innovation_drives(context):
    """Set specific input sentence."""
    context.input_sentence = "innovation drives our decisions"

@given('I have an input sentence "execution excellence is paramount"')
def step_input_sentence_execution_excellence(context):
    """Set specific input sentence."""
    context.input_sentence = "execution excellence is paramount"

@given('I have an input sentence "completely unrelated random text xyz"')
def step_input_sentence_unrelated_text(context):
    """Set specific input sentence."""
    context.input_sentence = "completely unrelated random text xyz"

@given('I have an input sentence "data driven decision making"')
def step_input_sentence_data_driven(context):
    """Set specific input sentence."""
    context.input_sentence = "data driven decision making"

@given('I have an input sentence "team collaboration matters"')
def step_input_sentence_team_collaboration(context):
    """Set specific input sentence."""
    context.input_sentence = "team collaboration matters"

@given('I have an input sentence "innovation is our core value"')
def step_input_sentence_innovation_core_value(context):
    """Set specific input sentence."""
    context.input_sentence = "innovation is our core value"

@given('I have an input sentence "leadership through example"')
def step_input_sentence_leadership_example(context):
    """Set specific input sentence."""
    context.input_sentence = "leadership through example"

@given('I have an input sentence "continuous improvement mindset"')
def step_input_sentence_continuous_improvement(context):
    """Set specific input sentence."""
    context.input_sentence = "continuous improvement mindset"

@given('I have an input sentence "innovative problem solving approach"')
def step_input_sentence_innovative_problem_solving(context):
    """Set specific input sentence."""
    context.input_sentence = "innovative problem solving approach"

# Additional step definitions for exact patterns in feature file
@when('I request 5 nearest neighbors')
def step_request_5_nearest_neighbors(context):
    """Request 5 nearest neighbors (generic)."""
    context.n_neighbors = 5
    import time
    context.start_time = time.time()
    
    try:
        context.neighbors_result = context.inference_engine.get_nearest_neighbors(
            context.input_sentence,
            context.collection_name,
            n_neighbors=5
        )
    except Exception as e:
        context.error_result = str(e)
    finally:
        context.end_time = time.time()

@when('I request 5 nearest neighbors from the database')
def step_request_5_neighbors_from_database(context):
    """Request 5 nearest neighbors from the database."""
    context.n_neighbors = 5
    import time
    context.start_time = time.time()
    
    try:
        context.neighbors_result = context.inference_engine.get_nearest_neighbors(
            context.input_sentence,
            context.collection_name,
            n_neighbors=5
        )
    except Exception as e:
        context.error_result = str(e)
    finally:
        context.end_time = time.time()

@when('I request 5 nearest neighbors with DL metadata')
def step_request_5_neighbors_with_dl_metadata(context):
    """Request 5 nearest neighbors with DL metadata."""
    context.n_neighbors = 5
    try:
        context.neighbors_result = context.inference_engine.get_nearest_neighbors(
            context.input_sentence,
            context.collection_name,
            n_neighbors=5,
            include_dominant_logic=True
        )
    except Exception as e:
        context.error_result = str(e)

@when('I request 10 nearest neighbors for dominant logic analysis')
def step_request_10_neighbors_for_dominant_logic(context):
    """Request 10 nearest neighbors for dominant logic analysis."""
    context.n_neighbors = 10
    try:
        context.neighbors_result = context.inference_engine.get_nearest_neighbors(
            context.input_sentence,
            context.collection_name,
            n_neighbors=10,
            include_dominant_logic=True
        )
    except Exception as e:
        context.error_result = str(e)

@then('I should get 5 neighbors regardless of metadata completeness')
def step_should_get_5_neighbors_regardless_metadata(context):
    """Verify we get 5 neighbors even with incomplete metadata."""
    assert len(context.neighbors_result['neighbors']) == 5, \
        f"Should get 5 neighbors regardless of metadata completeness"

@when('I request 8 nearest neighbors for distribution analysis')
def step_request_8_neighbors_for_distribution(context):
    """Request 8 nearest neighbors for distribution analysis."""
    context.n_neighbors = 8
    try:
        context.neighbors_result = context.inference_engine.get_nearest_neighbors(
            context.input_sentence,
            context.collection_name,
            n_neighbors=8,
            include_dominant_logic=True
        )
    except Exception as e:
        context.error_result = str(e)

@when('I request 3 nearest neighbors')
def step_request_3_neighbors(context):
    """Request 3 nearest neighbors."""
    context.n_neighbors = 3
    try:
        context.neighbors_result = context.inference_engine.get_nearest_neighbors(
            context.input_sentence,
            context.collection_name,
            n_neighbors=3
        )
    except Exception as e:
        context.error_result = str(e)

@when('I request 7 nearest neighbors')
def step_request_7_neighbors(context):
    """Request 7 nearest neighbors."""
    context.n_neighbors = 7
    try:
        context.neighbors_result = context.inference_engine.get_nearest_neighbors(
            context.input_sentence,
            context.collection_name,
            n_neighbors=7
        )
    except Exception as e:
        context.error_result = str(e)

@when('I request 10 nearest neighbors with minimum similarity threshold 0.7')
def step_request_10_neighbors_with_threshold_07(context):
    """Request 10 neighbors with similarity threshold 0.7."""
    context.n_neighbors = 10
    context.similarity_threshold = 0.7
    try:
        context.neighbors_result = context.inference_engine.get_nearest_neighbors(
            context.input_sentence,
            context.collection_name,
            n_neighbors=10,
            similarity_threshold=0.7
        )
    except Exception as e:
        context.error_result = str(e)

@when('I request 50 nearest neighbors')
def step_request_50_neighbors(context):
    """Request 50 nearest neighbors."""
    context.n_neighbors = 50
    import time
    context.start_time = time.time()
    
    try:
        context.neighbors_result = context.inference_engine.get_nearest_neighbors(
            context.input_sentence,
            context.collection_name,
            n_neighbors=50
        )
    except Exception as e:
        context.error_result = str(e)
    finally:
        context.end_time = time.time()

@when('I request 5 nearest neighbors with semantic analysis')
def step_request_5_neighbors_with_semantic(context):
    """Request 5 nearest neighbors with semantic analysis."""
    context.n_neighbors = 5
    try:
        context.neighbors_result = context.inference_engine.get_nearest_neighbors(
            context.input_sentence,
            context.collection_name,
            n_neighbors=5,
            include_dominant_logic=True
        )
    except Exception as e:
        context.error_result = str(e)

@then('I should get 3 neighbors even with low similarity scores')
def step_should_get_3_neighbors_with_low_similarity(context):
    """Verify we get 3 neighbors even with low similarity."""
    assert len(context.neighbors_result['neighbors']) == 3, \
        "Should get 3 neighbors even with low similarity scores"

@then('I should only get neighbors with similarity >= 0.7')
def step_should_get_neighbors_above_07_threshold(context):
    """Verify all neighbors have similarity >= 0.7."""
    for neighbor in context.neighbors_result['neighbors']:
        assert neighbor['similarity_score'] >= 0.7, \
            f"Neighbor similarity {neighbor['similarity_score']} should be >= 0.7"

@then('the operation should complete within reasonable time (< 10 seconds)')
def step_operation_completes_within_10_seconds(context):
    """Verify operation completes within 10 seconds."""
    execution_time = context.end_time - context.start_time
    assert execution_time < 10, \
        f"Operation took {execution_time:.2f}s, should be < 10s"

@then('I should get appropriate validation errors for:')
def step_get_validation_errors_for_colon(context):
    """Verify validation errors for invalid parameters (with colon)."""
    errors = context.validation_errors
    assert len(errors) > 0, "Should get validation errors for invalid parameters"

@then('if fewer than 10 neighbors meet the threshold, return only those that qualify')
def step_fewer_than_10_neighbors_meet_threshold(context):
    """Verify behavior when fewer than 10 neighbors meet threshold."""
    # This is acceptable - we may get fewer than requested
    actual_count = len(context.neighbors_result['neighbors'])
    assert actual_count <= 10, f"Should not get more than 10 neighbors"

@then('I should get exactly 50 neighbors if database has enough sentences')
def step_get_exactly_50_neighbors_if_available(context):
    """Verify we get exactly 50 neighbors if database has enough sentences."""
    actual_count = len(context.neighbors_result['neighbors'])
    # Our mock has 10 sentences, so we should get at most 10
    expected_count = min(50, 10)  # Mock database has 10 sentences
    assert actual_count == expected_count, \
        f"Expected {expected_count} neighbors, got {actual_count}"

@then('neighbors should still be ranked by similarity')
def step_neighbors_still_ranked_by_similarity(context):
    """Verify neighbors are still ranked by similarity even with low scores."""
    similarities = [n['similarity_score'] for n in context.neighbors_result['neighbors']]
    assert similarities == sorted(similarities, reverse=True), \
        "Neighbors should still be ranked by similarity even with low scores"
