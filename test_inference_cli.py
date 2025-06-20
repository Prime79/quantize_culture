#!/usr/bin/env python3
"""
Test for the inference CLI script.
Tests that the inference.py script correctly processes sentences and returns DL archetype classifications.
"""

import subprocess
import json
import sys
import os
from pathlib import Path

def test_inference_cli():
    """Test the inference CLI with various inputs."""
    
    # Test data
    test_cases = [
        {
            "sentence": "order is the key for success",
            "collection": "extended_contextualized_collection",
            "expected_fields": ["sentence", "cluster_id", "similarity_score", "classification", "confidence_level"]
        },
        {
            "sentence": "fail fast learn faster",
            "collection": "extended_contextualized_collection", 
            "expected_fields": ["sentence", "cluster_id", "similarity_score", "classification", "confidence_level"]
        }
    ]
    
    script_path = Path(__file__).parent / "inference.py"
    
    for i, test_case in enumerate(test_cases):
        print(f"\nTest {i+1}: Testing sentence '{test_case['sentence']}'")
        
        # Run the inference script
        try:
            result = subprocess.run([
                sys.executable, str(script_path),
                "--sentence", test_case["sentence"],
                "--collection", test_case["collection"],
                "--format", "json"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                print(f"ERROR: Script failed with return code {result.returncode}")
                print(f"STDERR: {result.stderr}")
                continue
                
            # Parse JSON output
            try:
                output = json.loads(result.stdout)
                print(f"✓ Successfully got JSON response")
                
                # Check required fields
                missing_fields = [field for field in test_case["expected_fields"] if field not in output]
                if missing_fields:
                    print(f"✗ Missing fields: {missing_fields}")
                else:
                    print(f"✓ All required fields present")
                    
                # Print results
                print(f"  - Cluster: {output.get('cluster_id', 'N/A')}")
                print(f"  - Similarity: {output.get('similarity_score', 'N/A'):.4f}")
                print(f"  - Classification: {output.get('classification', 'N/A')}")
                print(f"  - Confidence: {output.get('confidence_level', 'N/A')}")
                
            except json.JSONDecodeError as e:
                print(f"✗ Failed to parse JSON output: {e}")
                print(f"Raw output: {result.stdout}")
                
        except subprocess.TimeoutExpired:
            print(f"✗ Script timed out after 30 seconds")
        except Exception as e:
            print(f"✗ Error running script: {e}")

def test_inference_cli_help():
    """Test that the CLI shows help when requested."""
    script_path = Path(__file__).parent / "inference.py"
    
    result = subprocess.run([sys.executable, str(script_path), "--help"], 
                          capture_output=True, text=True)
    
    if result.returncode == 0 and "usage:" in result.stdout.lower():
        print("✓ Help command works")
    else:
        print("✗ Help command failed")
        print(f"Output: {result.stdout}")

if __name__ == "__main__":
    print("Testing inference CLI...")
    test_inference_cli_help()
    test_inference_cli()
    print("\nTest completed!")
