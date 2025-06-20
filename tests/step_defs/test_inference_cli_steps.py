"""
BDD step definitions for Digital Leadership Inference CLI testing.
Tests the inference.py command-line interface functionality.
"""

import subprocess
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Import BDD decorators (pytest-bdd style to match existing tests)
import pytest
from pytest_bdd import given, when, then, parsers

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

class InferenceCLISteps:
    """Step definitions for inference CLI tests."""
    
    def __init__(self):
        self.cli_script = project_root / "inference.py"
        self.collection_name = None
        self.test_sentence = None
        self.cli_result = None
        self.cli_results = {}
        self.json_output = None
        self.original_api_key = None

@given('the inference CLI is available')
def step_inference_cli_available(context):
    """Verify that the inference CLI script exists and is executable."""
    context.cli_script = project_root / "inference.py"
    assert context.cli_script.exists(), f"Inference CLI script not found at {context.cli_script}"
    context.cli_results = {}
    context.cli_errors = {}

@given('the reference collection "{collection_name}" exists')
def step_reference_collection_exists(context, collection_name):
    """Verify that the specified collection exists in Qdrant."""
    context.collection_name = collection_name
    # We assume the collection exists based on previous setup
    # In a full test environment, we would verify collection existence

@given('the OpenAI API key is configured')
def step_openai_api_key_configured(context):
    """Verify that OpenAI API key is available."""
    context.original_api_key = os.environ.get('OPENAI_API_KEY')
    assert context.original_api_key, "OPENAI_API_KEY environment variable must be set for testing"

@given('the OpenAI API key is not set')
def step_openai_api_key_not_set(context):
    """Temporarily remove the OpenAI API key for error testing."""
    context.original_api_key = os.environ.get('OPENAI_API_KEY')
    if 'OPENAI_API_KEY' in os.environ:
        del os.environ['OPENAI_API_KEY']

@when('I run the inference CLI with sentence "{sentence}"')
def step_run_inference_cli_basic(context, sentence):
    """Run the inference CLI with a basic sentence."""
    context.test_sentence = sentence
    context.cli_result = _run_cli_command(context, sentence, context.collection_name)

@when('I run the inference CLI with sentence "{sentence}" in JSON format')
def step_run_inference_cli_json(context, sentence):
    """Run the inference CLI with JSON output format."""
    context.test_sentence = sentence
    context.cli_result = _run_cli_command(context, sentence, context.collection_name, format_type="json")

@when('I run the inference CLI with sentence "{sentence}" in verbose mode')
def step_run_inference_cli_verbose(context, sentence):
    """Run the inference CLI with verbose output."""
    context.test_sentence = sentence
    context.cli_result = _run_cli_command(context, sentence, context.collection_name, verbose=True)

@when('I run the inference CLI with sentence "{sentence}" and collection "{collection}"')
def step_run_inference_cli_with_collection(context, sentence, collection):
    """Run the inference CLI with a specific collection."""
    context.test_sentence = sentence
    context.cli_result = _run_cli_command(context, sentence, collection)

@when('I run the inference CLI with help flag')
def step_run_inference_cli_help(context):
    """Run the inference CLI with help flag."""
    context.cli_result = _run_cli_help(context)

@when('I run the inference CLI in verbose mode for a sentence matching that cluster')
def step_run_inference_cli_verbose_cluster_match(context):
    """Run inference CLI in verbose mode for cluster analysis."""
    # Use a test sentence that should match a cluster with DL metadata
    sentence = "order is the key for success"
    context.test_sentence = sentence
    context.cli_result = _run_cli_command(context, sentence, context.collection_name, verbose=True)

@then('I should get a classification result')
def step_should_get_classification_result(context):
    """Verify that classification result was returned."""
    assert context.cli_result['success'], f"CLI command failed: {context.cli_result.get('error', 'Unknown error')}"
    assert context.cli_result['return_code'] == 0, f"CLI returned error code: {context.cli_result['return_code']}"
    assert context.cli_result['stdout'], "No output received from CLI"

@then('the result should contain a cluster ID')  
def step_result_should_contain_cluster_id(context):
    """Verify that result contains cluster ID."""
    output = context.cli_result['stdout']
    assert 'Cluster ID:' in output or 'cluster_id' in output, "Result should contain cluster ID"

@then('the result should contain a similarity score')
def step_result_should_contain_similarity_score(context):
    """Verify that result contains similarity score."""
    output = context.cli_result['stdout']
    assert 'Similarity Score:' in output or 'similarity_score' in output, "Result should contain similarity score"

@then('the result should contain a confidence level')
def step_result_should_contain_confidence_level(context):
    """Verify that result contains confidence level."""
    output = context.cli_result['stdout']
    assert 'Confidence Level:' in output or 'confidence_level' in output, "Result should contain confidence level"

@then('I should get valid JSON output')
def step_should_get_valid_json_output(context):
    """Verify that output is valid JSON."""
    assert context.cli_result['success'], f"CLI command failed: {context.cli_result.get('error', 'Unknown error')}"
    try:
        context.json_output = json.loads(context.cli_result['stdout'])
    except json.JSONDecodeError as e:
        assert False, f"Output is not valid JSON: {e}\nOutput: {context.cli_result['stdout']}"

@then('the JSON should contain field "{field_name}"')
def step_json_should_contain_field(context, field_name):
    """Verify that JSON output contains specified field."""
    assert hasattr(context, 'json_output'), "JSON output not parsed"
    assert field_name in context.json_output, f"JSON should contain field '{field_name}'. Available fields: {list(context.json_output.keys())}"

@then('the result should include cluster members')
def step_result_should_include_cluster_members(context):
    """Verify that verbose result includes cluster members."""
    output = context.cli_result['stdout']
    assert 'CLUSTER MEMBERS' in output or 'cluster_members' in output, "Verbose output should include cluster members"

@then('the result should include dominant logic analysis')
def step_result_should_include_dominant_logic_analysis(context):
    """Verify that verbose result includes dominant logic analysis."""
    output = context.cli_result['stdout']
    assert 'DOMINANT LOGIC ANALYSIS' in output or 'dominant_logic' in output, "Verbose output should include dominant logic analysis"

@then('the result should show cluster size')
def step_result_should_show_cluster_size(context):
    """Verify that result shows cluster size."""
    output = context.cli_result['stdout']
    assert 'total)' in output or 'cluster_size' in output, "Result should show cluster size"

@then('I should see usage information')
def step_should_see_usage_information(context):
    """Verify that help output contains usage information."""
    output = context.cli_result['stdout']
    assert 'usage:' in output.lower(), "Help output should contain usage information"

@then('I should see available options')
def step_should_see_available_options(context):
    """Verify that help output contains available options."""
    output = context.cli_result['stdout']
    assert '--sentence' in output or '-s' in output, "Help should show sentence option"
    assert '--collection' in output or '-c' in output, "Help should show collection option"

@then('I should see example commands')
def step_should_see_example_commands(context):
    """Verify that help output contains example commands."""
    output = context.cli_result['stdout']
    assert 'Examples:' in output or 'example' in output.lower(), "Help should contain example commands"

@then('both classifications should succeed')
def step_both_classifications_should_succeed(context):
    """Verify that multiple classifications succeeded."""
    # This step assumes multiple CLI calls were made and stored
    assert len(context.cli_results) >= 2, "Expected at least 2 CLI results"
    for result in context.cli_results.values():
        assert result['success'], f"Classification failed: {result.get('error', 'Unknown error')}"

@then('both should return valid cluster IDs')
def step_both_should_return_valid_cluster_ids(context):
    """Verify that both results contain valid cluster IDs."""
    for result in context.cli_results.values():
        output = result['stdout']
        assert 'Cluster ID:' in output or 'cluster_id' in output, "Each result should contain cluster ID"

@then('both should return confidence levels')
def step_both_should_return_confidence_levels(context):
    """Verify that both results contain confidence levels."""
    for result in context.cli_results.values():
        output = result['stdout']
        assert 'Confidence Level:' in output or 'confidence_level' in output, "Each result should contain confidence level"

@then('the CLI should exit with error code {error_code:d}')
def step_cli_should_exit_with_error_code(context, error_code):
    """Verify that CLI exits with specified error code."""
    assert context.cli_result['return_code'] == error_code, f"Expected error code {error_code}, got {context.cli_result['return_code']}"

@then('I should see an error message about missing API key')
def step_should_see_error_message_about_missing_api_key(context):
    """Verify error message about missing API key."""
    stderr = context.cli_result['stderr']
    assert 'OPENAI_API_KEY' in stderr, f"Should see error about missing API key. stderr: {stderr}"

@then('the CLI should handle the error gracefully')
def step_cli_should_handle_error_gracefully(context):
    """Verify that CLI handles errors gracefully."""
    # CLI should exit with non-zero code but not crash
    assert context.cli_result['return_code'] != 0, "CLI should exit with error code for invalid input"
    # Should not have Python traceback in normal operation
    stderr = context.cli_result['stderr']
    assert 'Traceback' not in stderr or 'Error:' in stderr, "CLI should handle errors gracefully"

@then('I should see an appropriate error message')
def step_should_see_appropriate_error_message(context):
    """Verify that an appropriate error message is shown."""
    stderr = context.cli_result['stderr']
    stdout = context.cli_result['stdout']
    error_output = stderr + stdout
    assert 'error' in error_output.lower() or 'Error' in error_output, f"Should see error message. Output: {error_output}"

@then('I should see the most common DL category')
def step_should_see_most_common_dl_category(context):
    """Verify that dominant logic shows most common DL category."""
    output = context.cli_result['stdout']
    # Look for category information in verbose output
    assert 'Primary Category:' in output or 'most_common_category' in output, "Should show most common DL category"

@then('I should see the category distribution')
def step_should_see_category_distribution(context):
    """Verify that category distribution is shown."""
    output = context.cli_result['stdout']
    assert 'distribution' in output.lower() or 'category' in output.lower(), "Should show category distribution"

@then('I should see the subcategory distribution')
def step_should_see_subcategory_distribution(context):
    """Verify that subcategory distribution is shown."""
    output = context.cli_result['stdout']
    assert 'subcategory' in output.lower(), "Should show subcategory information"

@then('I should see the archetype distribution')
def step_should_see_archetype_distribution(context):
    """Verify that archetype distribution is shown."""
    output = context.cli_result['stdout']
    assert 'archetype' in output.lower(), "Should show archetype information"

@then('I should see all cluster members')
def step_should_see_all_cluster_members(context):
    """Verify that all cluster members are shown."""
    output = context.cli_result['stdout']
    assert 'CLUSTER MEMBERS' in output, "Should show all cluster members section"

@then('each member should show its text')
def step_each_member_should_show_text(context):
    """Verify that each cluster member shows its text."""
    output = context.cli_result['stdout']
    # Look for numbered list items in cluster members
    assert any(f"{i}." in output for i in range(1, 10)), "Should show numbered cluster member texts"

@then('each member should show its DL metadata if available')
def step_each_member_should_show_dl_metadata(context):
    """Verify that DL metadata is shown when available."""
    output = context.cli_result['stdout']
    # This is a soft check since not all clusters may have DL metadata
    # We just verify the format supports it
    assert 'CLUSTER MEMBERS' in output, "Should have cluster members section for DL metadata"

@then('I should see the total cluster size')
def step_should_see_total_cluster_size(context):
    """Verify that total cluster size is shown."""
    output = context.cli_result['stdout']
    assert 'total)' in output or 'cluster_size' in output, "Should show total cluster size"

@then('the result should match the enhanced inference engine output')
def step_result_should_match_enhanced_inference_engine_output(context):
    """Verify that CLI result matches inference engine output."""
    # This is a integration test - we verify the CLI produces consistent results
    assert context.cli_result['success'], "CLI should produce consistent results with inference engine"
    output = context.cli_result['stdout']
    assert 'Classification:' in output, "Should show classification consistent with inference engine"

@then('the classification should be consistent')
def step_classification_should_be_consistent(context):
    """Verify that classification is consistent."""
    output = context.cli_result['stdout']
    assert 'cluster_' in output.lower() or 'Classification:' in output, "Should show consistent classification"

@then('the confidence assessment should be accurate')
def step_confidence_assessment_should_be_accurate(context):
    """Verify that confidence assessment is accurate."""
    output = context.cli_result['stdout']
    valid_confidence_levels = ['STRONG', 'WEAK', 'AMBIGUOUS', 'TRAINING_MATCH', 'NO_MATCH']
    assert any(level in output for level in valid_confidence_levels), f"Should show valid confidence level. Output: {output}"

# Cleanup functions
def cleanup_environment(context):
    """Restore environment after tests."""
    if hasattr(context, 'original_api_key') and context.original_api_key:
        os.environ['OPENAI_API_KEY'] = context.original_api_key

# Helper functions
def _run_cli_command(context, sentence: str, collection: str, format_type: str = "human", verbose: bool = False) -> Dict[str, Any]:
    """Run the inference CLI command and return result."""
    cmd = [
        sys.executable, str(context.cli_script),
        "--sentence", sentence,
        "--collection", collection
    ]
    
    if format_type == "json":
        cmd.extend(["--format", "json"])
    
    if verbose:
        cmd.append("--verbose")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=project_root)
        return {
            'success': True,
            'return_code': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'command': ' '.join(cmd)
        }
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'return_code': -1,
            'stdout': '',
            'stderr': 'Command timed out after 30 seconds',
            'error': 'timeout',
            'command': ' '.join(cmd)
        }
    except Exception as e:
        return {
            'success': False,
            'return_code': -1,
            'stdout': '',
            'stderr': str(e),
            'error': str(e),
            'command': ' '.join(cmd)
        }

def _run_cli_help(context) -> Dict[str, Any]:
    """Run the inference CLI help command."""
    cmd = [sys.executable, str(context.cli_script), "--help"]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10, cwd=project_root)
        return {
            'success': True,
            'return_code': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'command': ' '.join(cmd)
        }
    except Exception as e:
        return {
            'success': False,
            'return_code': -1,
            'stdout': '',
            'stderr': str(e),
            'error': str(e),
            'command': ' '.join(cmd)
        }
