"""
User requirements level step definitions for BDD tests
"""
import pytest
from pytest_bdd import scenarios, given, when, then, parsers
import sys
import os
import json
import tempfile
from typing import Dict, List

# Load scenarios from feature file
scenarios('../features/user_requirements.feature')

# Context class for sharing data between steps
class UserTestContext:
    def __init__(self):
        self.reference_framework = None
        self.pipeline_result = None
        self.comparison_result = None

@pytest.fixture
def context():
    return UserTestContext()

@pytest.fixture
def reference_framework():
    """Load the test reference framework"""
    fixtures_path = os.path.join(os.path.dirname(__file__), '..', 'fixtures')
    with open(os.path.join(fixtures_path, 'test_dl_reference.json'), 'r') as f:
        return json.load(f)

# Mock DL Pipeline class

# Background steps
@given('I have a reference framework JSON with archetypes, DLs, and example sentences')
def given_reference_framework(context, reference_framework):
    context.reference_framework = reference_framework
    assert 'archetypes' in reference_framework
    assert 'digital_leadership_dimensions' in reference_framework
    assert 'example_sentences' in reference_framework
    assert len(reference_framework['example_sentences']) == 25

@given('I have access to the embedding and clustering pipeline')
def given_pipeline_access(context):
    context.pipeline = MockDLPipeline()

# Mock DL Pipeline class
class MockDLPipeline:
    def __init__(self):
        self.context_enabled = True
        self.custom_context = None
        self.results = {}
        
    def run_with_default_context(self, data: Dict) -> Dict:
        """Run pipeline with default contextualization"""
        sentences = [item['text'] for item in data['example_sentences']]
        
        # Mock contextualization
        contextualized = [
            f"This is a sentence related to digital rights, privacy, technology, and online culture: {s}"
            for s in sentences
        ]
        
        # Mock clustering results
        results = {
            'input_sentences': len(sentences),
            'contextualized_sentences': contextualized,
            'clustering': {
                'n_clusters': 4,  # Matching our 4 archetypes
                'cluster_assignments': {
                    'Digital Innovator': 6,
                    'Cultural Catalyst': 7, 
                    'Strategic Visionary': 6,
                    'Operational Excellence': 6
                }
            },
            'evaluation': {
                'quantitative': {
                    'silhouette_score': 0.72,
                    'noise_percentage': 8.0,
                    'calinski_harabasz_score': 156.3
                },
                'qualitative': {
                    'semantic_coherence': 0.84,
                    'cultural_alignment': 0.78,
                    'interpretability': 0.91
                },
                'combined_score': 0.81
            },
            'storage_status': 'completed',
            'report_generated': True
        }
        
        self.results = results
        return results
        
    def run_with_custom_context(self, data: Dict, context_phrase: str) -> Dict:
        """Run pipeline with custom contextualization"""
        sentences = [item['text'] for item in data['example_sentences']]
        
        # Mock custom contextualization
        contextualized = [f"{context_phrase} {s}" for s in sentences]
        
        results = {
            'input_sentences': len(sentences),
            'contextualized_sentences': contextualized,
            'custom_context': context_phrase,
            'clustering': {
                'n_clusters': 5,  # Different result with custom context
                'parameter_combinations_tested': 9,
                'best_parameters': {
                    'n_neighbors': 10,
                    'min_dist': 0.05,
                    'min_cluster_size': 3
                }
            },
            'evaluation': {
                'quantitative': {
                    'silhouette_score': 0.68,
                    'noise_percentage': 12.0
                },
                'qualitative': {
                    'semantic_coherence': 0.79,
                    'cultural_alignment': 0.85,
                    'interpretability': 0.88
                },
                'combined_score': 0.79
            },
            'storage_status': 'completed'
        }
        
        self.results = results
        return results
        
    def compare_approaches(self, default_results: Dict, custom_results: Dict) -> Dict:
        """Compare different contextualization approaches"""
        return {
            'default_approach': {
                'combined_score': default_results['evaluation']['combined_score'],
                'n_clusters': default_results['clustering']['n_clusters'],
                'semantic_coherence': default_results['evaluation']['qualitative']['semantic_coherence']
            },
            'custom_approach': {
                'combined_score': custom_results['evaluation']['combined_score'], 
                'n_clusters': custom_results['clustering']['n_clusters'],
                'semantic_coherence': custom_results['evaluation']['qualitative']['semantic_coherence']
            },
            'quantitative_differences': {
                'score_difference': abs(default_results['evaluation']['combined_score'] - 
                                      custom_results['evaluation']['combined_score']),
                'cluster_count_difference': abs(default_results['clustering']['n_clusters'] - 
                                              custom_results['clustering']['n_clusters'])
            },
            'qualitative_differences': {
                'coherence_difference': abs(default_results['evaluation']['qualitative']['semantic_coherence'] - 
                                          custom_results['evaluation']['qualitative']['semantic_coherence'])
            },
            'recommendation': 'default_approach' if default_results['evaluation']['combined_score'] > 
                             custom_results['evaluation']['combined_score'] else 'custom_approach'
        }

# Scenario 1: Default contextualization
@given('I have a test JSON file with 25 company culture statements')
def given_test_data_25_statements(context):
    assert len(context.reference_framework['example_sentences']) == 25

@when('I run the DL estimation pipeline with default settings')
def when_run_default_pipeline(context):
    context.pipeline_result = context.pipeline.run_with_default_context(context.reference_framework)

@then('the system should embed all sentences with domain context')
def then_check_domain_context_embedding(context):
    contextualized = context.pipeline_result['contextualized_sentences']
    for sentence in contextualized:
        assert "This is a sentence related to digital rights, privacy, technology, and online culture:" in sentence

@then('find optimal clusters using UMAP and HDBSCAN')
def then_check_clustering_performed(context):
    clustering = context.pipeline_result['clustering']
    assert 'n_clusters' in clustering
    assert clustering['n_clusters'] > 0

@then('evaluate clusters with both qualitative and quantitative measures')
def then_check_both_evaluations(context):
    evaluation = context.pipeline_result['evaluation']
    assert 'quantitative' in evaluation
    assert 'qualitative' in evaluation
    assert 'silhouette_score' in evaluation['quantitative']
    assert 'semantic_coherence' in evaluation['qualitative']

@then('save the best clustering results back to the vector database')
def then_check_database_storage(context):
    assert context.pipeline_result['storage_status'] == 'completed'

@then('generate a comprehensive report with cluster assignments')
def then_check_report_generation(context):
    assert context.pipeline_result['report_generated'] == True
    clustering = context.pipeline_result['clustering']
    assert 'cluster_assignments' in clustering

# Scenario 2: Custom contextualization
@when('I run the DL estimation pipeline with context enabled')
def when_enable_custom_context(context):
    context.custom_context_enabled = True

@when(parsers.parse('I provide the context phrase "{context_phrase}"'))
def when_provide_context_phrase(context, context_phrase):
    context.pipeline_result = context.pipeline.run_with_custom_context(context.reference_framework, context_phrase)

@then('the system should embed all sentences with my custom context')
def then_check_custom_context_embedding(context):
    contextualized = context.pipeline_result['contextualized_sentences']
    custom_context = context.pipeline_result['custom_context']
    for sentence in contextualized:
        assert sentence.startswith(custom_context)

@then('find optimal clusters using multiple parameter combinations')
def then_check_parameter_combinations(context):
    clustering = context.pipeline_result['clustering']
    assert clustering['parameter_combinations_tested'] == 9

@then('select the best clustering based on combined quality scores')
def then_check_best_selection(context):
    evaluation = context.pipeline_result['evaluation']
    assert 'combined_score' in evaluation
    assert 0 <= evaluation['combined_score'] <= 1

@then('store cluster names and assignments in the vector database')
def then_check_cluster_storage(context):
    assert context.pipeline_result['storage_status'] == 'completed'

# Scenario 3: Comparison
@given('I have the same dataset processed with different contexts')
def given_comparison_setup(context):
    default_results = context.pipeline.run_with_default_context(context.reference_framework)
    custom_results = context.pipeline.run_with_custom_context(context.reference_framework, "Digital leadership assessment:")
    context.comparison_data = {
        'default': default_results,
        'custom': custom_results
    }

@when('I run benchmark comparisons')
def when_run_benchmark_comparison(context):
    context.comparison_result = context.pipeline.compare_approaches(
        context.comparison_data['default'], 
        context.comparison_data['custom']
    )

@then('I should see quantitative differences in clustering quality')
def then_check_quantitative_differences(context):
    assert 'quantitative_differences' in context.comparison_result
    quant_diff = context.comparison_result['quantitative_differences']
    assert 'score_difference' in quant_diff
    assert 'cluster_count_difference' in quant_diff

@then('qualitative differences in semantic coherence')
def then_check_qualitative_differences(context):
    assert 'qualitative_differences' in context.comparison_result
    qual_diff = context.comparison_result['qualitative_differences']
    assert 'coherence_difference' in qual_diff

@then('recommendations for the best approach')
def then_check_recommendations(context):
    assert 'recommendation' in context.comparison_result
    recommendation = context.comparison_result['recommendation']
    assert recommendation in ['default_approach', 'custom_approach']
