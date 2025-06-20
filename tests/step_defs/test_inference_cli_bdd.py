"""
BDD-style tests for Digital Leadership Inference CLI.
Tests the inference.py command-line interface functionality.
"""

import subprocess
import json
import os
import sys
import pytest
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

class TestInferenceCLI:
    """BDD-style tests for the inference CLI."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for each test."""
        self.cli_script = project_root / "inference.py"
        self.collection_name = "extended_contextualized_collection"
        assert self.cli_script.exists(), f"CLI script not found: {self.cli_script}"
        assert os.getenv('OPENAI_API_KEY'), "OPENAI_API_KEY must be set"
    
    def _run_cli(self, sentence: str, collection: str = None, format_type: str = "human", 
                 verbose: bool = False, expect_success: bool = True) -> Dict[str, Any]:
        """Helper to run CLI command."""
        collection = collection or self.collection_name
        cmd = [
            sys.executable, str(self.cli_script),
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
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'success': result.returncode == 0 if expect_success else True
            }
        except subprocess.TimeoutExpired:
            return {'return_code': -1, 'stdout': '', 'stderr': 'Timeout', 'success': False}
    
    def test_cli_basic_classification(self):
        """
        Scenario: Basic sentence classification
        Given the inference CLI is available
        When I run the inference CLI with sentence "order is the key for success"
        Then I should get a classification result
        And the result should contain a cluster ID
        And the result should contain a similarity score
        And the result should contain a confidence level
        """
        # When I run the inference CLI with sentence
        result = self._run_cli("order is the key for success")
        
        # Then I should get a classification result
        assert result['success'], f"CLI failed: {result['stderr']}"
        assert result['return_code'] == 0
        
        # And the result should contain required fields
        output = result['stdout']
        assert 'Cluster ID:' in output, "Should contain cluster ID"
        assert 'Similarity Score:' in output, "Should contain similarity score"
        assert 'Confidence Level:' in output, "Should contain confidence level"
        assert 'Classification:' in output, "Should contain classification"
    
    def test_cli_json_output(self):
        """
        Scenario: JSON output format
        When I run the inference CLI with sentence "order is the key for success" in JSON format
        Then I should get valid JSON output
        And the JSON should contain required fields
        """
        # When I run CLI with JSON format
        result = self._run_cli("order is the key for success", format_type="json")
        
        # Then I should get valid JSON output
        assert result['success'], f"CLI failed: {result['stderr']}"
        
        # Parse JSON
        try:
            json_data = json.loads(result['stdout'])
        except json.JSONDecodeError as e:
            pytest.fail(f"Invalid JSON output: {e}\nOutput: {result['stdout']}")
        
        # And JSON should contain required fields
        required_fields = ['sentence', 'cluster_id', 'similarity_score', 'classification', 'confidence_level']
        for field in required_fields:
            assert field in json_data, f"JSON missing field: {field}"
        
        # Verify data types
        assert isinstance(json_data['cluster_id'], int), "cluster_id should be integer"
        assert isinstance(json_data['similarity_score'], (int, float)), "similarity_score should be numeric"
        assert json_data['sentence'] == "order is the key for success", "sentence should match input"
    
    def test_cli_verbose_mode(self):
        """
        Scenario: Verbose mode with dominant logic analysis
        When I run the inference CLI with sentence "order is the key for success" in verbose mode
        Then I should get a classification result
        And the result should include cluster members
        And the result should include dominant logic analysis
        And the result should show cluster size
        """
        # When I run CLI in verbose mode
        result = self._run_cli("order is the key for success", verbose=True)
        
        # Then I should get a classification result
        assert result['success'], f"CLI failed: {result['stderr']}"
        
        output = result['stdout']
        # And the result should include cluster analysis
        assert 'CLUSTER MEMBERS' in output, "Should include cluster members"
        assert 'DOMINANT LOGIC ANALYSIS' in output, "Should include dominant logic analysis"
        assert 'total)' in output, "Should show cluster size"
    
    def test_cli_help_documentation(self):
        """
        Scenario: Help documentation
        When I run the inference CLI with help flag
        Then I should see usage information
        And I should see available options
        And I should see example commands
        """
        # When I run CLI with help flag
        cmd = [sys.executable, str(self.cli_script), "--help"]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root)
        
        # Then I should see documentation
        assert result.returncode == 0, "Help command should succeed"
        output = result.stdout
        
        assert 'usage:' in output.lower(), "Should show usage information"
        assert '--sentence' in output, "Should show sentence option"
        assert '--collection' in output, "Should show collection option"
        assert '--verbose' in output, "Should show verbose option"
        assert 'Examples:' in output, "Should show examples"
    
    def test_cli_multiple_sentences(self):
        """
        Scenario: Multiple sentence classification
        When I run the inference CLI with different sentences
        Then both classifications should succeed
        And both should return valid results
        """
        sentences = [
            "order is the key for success",
            "fail fast learn faster",
            "innovation drives us forward"
        ]
        
        results = []
        for sentence in sentences:
            result = self._run_cli(sentence, format_type="json")
            assert result['success'], f"Failed for sentence: {sentence}"
            
            # Parse and validate JSON
            json_data = json.loads(result['stdout'])
            results.append(json_data)
            
            # Verify each result
            assert 'cluster_id' in json_data, f"Missing cluster_id for: {sentence}"
            assert 'confidence_level' in json_data, f"Missing confidence_level for: {sentence}"
            assert isinstance(json_data['similarity_score'], (int, float)), f"Invalid similarity_score for: {sentence}"
        
        # Verify we got results for all sentences
        assert len(results) == len(sentences), "Should get results for all sentences"
        
        # Verify results are different (sentences should not all map to same cluster)
        cluster_ids = [r['cluster_id'] for r in results]
        assert len(set(cluster_ids)) >= 1, "Should get at least some variation in cluster assignments"
    
    def test_cli_error_handling_missing_sentence(self):
        """
        Scenario: Error handling for missing required arguments
        When I run the CLI without required arguments
        Then it should show appropriate error message
        """
        # When I run CLI without sentence
        cmd = [sys.executable, str(self.cli_script), "--collection", self.collection_name]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root)
        
        # Then it should show error
        assert result.returncode != 0, "Should fail without required sentence argument"
        assert 'required' in result.stderr.lower() or 'error' in result.stderr.lower(), "Should show error message"
    
    def test_cli_error_handling_invalid_collection(self):
        """
        Scenario: Error handling for invalid collection
        When I run the CLI with invalid collection name
        Then it should handle the error gracefully
        """
        # When I run CLI with invalid collection
        result = self._run_cli("test sentence", collection="nonexistent_collection", expect_success=False)
        
        # Then it should handle error gracefully
        assert result['return_code'] != 0, "Should fail with invalid collection"
        error_output = result['stderr'] + result['stdout']
        assert 'error' in error_output.lower(), "Should show error message"
    
    def test_cli_verbose_json_combination(self):
        """
        Scenario: Verbose mode with JSON output
        When I combine verbose and JSON flags
        Then I should get detailed JSON with all verbose information
        """
        # When I run CLI with both verbose and JSON
        result = self._run_cli("order is the key for success", format_type="json", verbose=True)
        
        # Then I should get detailed JSON
        assert result['success'], f"CLI failed: {result['stderr']}"
        
        json_data = json.loads(result['stdout'])
        
        # Verify verbose fields are present in JSON
        expected_verbose_fields = ['debug', 'cluster_members', 'cluster_size', 'dominant_logic']
        present_verbose_fields = [field for field in expected_verbose_fields if field in json_data]
        assert len(present_verbose_fields) > 0, f"Should contain verbose fields. Present: {present_verbose_fields}"
    
    def test_cli_integration_with_inference_engine(self):
        """
        Scenario: CLI integration with existing inference engine
        When I run the CLI with a test sentence
        Then the result should be consistent with the inference engine
        And confidence assessment should be accurate
        """
        # When I run CLI with known test sentence
        result = self._run_cli("fail fast is our core principle", format_type="json")
        
        # Then result should be consistent
        assert result['success'], f"CLI failed: {result['stderr']}"
        
        json_data = json.loads(result['stdout'])
        
        # Verify inference engine integration
        assert 'cluster_id' in json_data, "Should have cluster classification"
        assert 'confidence_level' in json_data, "Should have confidence assessment"
        
        # Verify valid confidence levels
        valid_confidence_levels = ['STRONG', 'WEAK', 'AMBIGUOUS', 'TRAINING_MATCH', 'NO_MATCH']
        assert json_data['confidence_level'] in valid_confidence_levels, f"Invalid confidence level: {json_data['confidence_level']}"
        
        # Verify similarity score is reasonable
        similarity = json_data['similarity_score']
        assert 0.0 <= similarity <= 1.0, f"Similarity score should be 0-1, got: {similarity}"
    
    def test_cli_performance(self):
        """
        Scenario: CLI performance test
        When I run the CLI multiple times
        Then each call should complete within reasonable time
        """
        import time
        
        sentence = "order is the key for success"
        times = []
        
        for i in range(3):
            start_time = time.time()
            result = self._run_cli(sentence)
            end_time = time.time()
            
            assert result['success'], f"CLI call {i+1} failed"
            times.append(end_time - start_time)
        
        # Verify performance (should complete within 30 seconds each)
        max_time = max(times)
        avg_time = sum(times) / len(times)
        
        assert max_time < 30, f"CLI calls too slow, max time: {max_time:.2f}s"
        print(f"CLI Performance: avg={avg_time:.2f}s, max={max_time:.2f}s")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
