#!/usr/bin/env python3
"""
Test runner script for Digital Leadership Assessment BDD tests
"""
import os
import sys
import subprocess
from pathlib import Path

def run_tests():
    """Run all BDD tests with proper reporting"""
    
    # Get the project root directory
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    print("🧪 Running Digital Leadership Assessment BDD Tests")
    print("=" * 60)
    
    # Run different test levels
    test_commands = [
        {
            'name': 'Unit Tests (Level 3)',
            'command': ['python', '-m', 'pytest', 'tests/step_defs/test_simple_unit.py', '-v', '--tb=short'],
            'marker': 'unit'
        },
        {
            'name': 'Functional Tests (Level 2)', 
            'command': ['python', '-m', 'pytest', 'tests/step_defs/test_functional_steps.py', '-v', '--tb=short'],
            'marker': 'functional'
        },
        {
            'name': 'User Requirements Tests (Level 1)',
            'command': ['python', '-m', 'pytest', 'tests/step_defs/test_user_requirements_steps.py', '-v', '--tb=short'],
            'marker': 'user'
        },
        {
            'name': 'Enhanced Inference Tests (Advanced)',
            'command': ['python', '-m', 'pytest', 'tests/step_defs/test_enhanced_inference_steps.py', '-v', '--tb=short'],
            'marker': 'enhanced'
        },
        {
            'name': 'Inference CLI Tests (Integration)',
            'command': ['python', '-m', 'pytest', 'tests/step_defs/test_inference_cli_bdd.py', '-v', '--tb=short'],
            'marker': 'cli'
        }
    ]
    
    all_passed = True
    results = []
    
    for test_suite in test_commands:
        print(f"\n🔬 Running {test_suite['name']}")
        print("-" * 40)
        
        try:
            result = subprocess.run(
                test_suite['command'],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                print(f"✅ {test_suite['name']}: PASSED")
                results.append(f"✅ {test_suite['name']}: PASSED")
            else:
                print(f"❌ {test_suite['name']}: FAILED")
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)
                results.append(f"❌ {test_suite['name']}: FAILED")
                all_passed = False
                
        except subprocess.TimeoutExpired:
            print(f"⏰ {test_suite['name']}: TIMEOUT")
            results.append(f"⏰ {test_suite['name']}: TIMEOUT")
            all_passed = False
        except Exception as e:
            print(f"💥 {test_suite['name']}: ERROR - {e}")
            results.append(f"💥 {test_suite['name']}: ERROR")
            all_passed = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    for result in results:
        print(result)
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! The DL Assessment pipeline is ready.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
        return 1

def run_specific_level(level: str):
    """Run tests for a specific BDD level"""
    level_mapping = {
        '1': 'tests/step_defs/test_user_requirements_steps.py',
        '2': 'tests/step_defs/test_functional_steps.py', 
        '3': 'tests/step_defs/test_simple_unit.py',
        'enhanced': 'tests/step_defs/test_enhanced_inference_steps.py',
        'cli': 'tests/step_defs/test_inference_cli_bdd.py'
    }
    
    if level not in level_mapping:
        print(f"❌ Invalid level: {level}. Use 1, 2, 3, 'enhanced', or 'cli'.")
        return 1
        
    test_file = level_mapping[level]
    level_names = {
        '1': 'User Requirements', 
        '2': 'Functional', 
        '3': 'Unit Tests',
        'enhanced': 'Enhanced Inference',
        'cli': 'Inference CLI'
    }
    
    print(f"🧪 Running Level {level}: {level_names[level]} Tests")
    print("=" * 50)
    
    try:
        result = subprocess.run([
            'python', '-m', 'pytest', test_file, '-v', '--tb=short'
        ], timeout=300)
        return result.returncode
    except Exception as e:
        print(f"💥 Error running tests: {e}")
        return 1

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Run specific level
        level = sys.argv[1]
        exit_code = run_specific_level(level)
    else:
        # Run all tests
        exit_code = run_tests()
    
    sys.exit(exit_code)
