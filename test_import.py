#!/usr/bin/env python3
"""Test script to debug the import issue."""

# Test 1: Direct import
try:
    from app.data.inference_models import EnhancedReferenceMetadata
    print("✓ Direct import successful")
except Exception as e:
    print("✗ Direct import failed:", e)

# Test 2: Star import (like in inference_engine.py)
try:
    from app.data.inference_models import *
    print("✓ Star import successful")
    print("✓ EnhancedReferenceMetadata available:", 'EnhancedReferenceMetadata' in globals())
except Exception as e:
    print("✗ Star import failed:", e)

# Test 3: Module import
try:
    import app.data.inference_models as models
    print("✓ Module import successful")
    print("✓ EnhancedReferenceMetadata in module:", hasattr(models, 'EnhancedReferenceMetadata'))
    if hasattr(models, 'EnhancedReferenceMetadata'):
        print("✓ Can access class:", models.EnhancedReferenceMetadata)
except Exception as e:
    print("✗ Module import failed:", e)
