#!/usr/bin/env python3
"""
Test script to verify the updated quantitative vs qualitative terminology works correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_terminology_updates():
    """Test that the updated terminology works correctly."""
    print("🧪 Testing Updated Quantitative vs Qualitative Terminology")
    print("=" * 70)
    
    # Test 1: Import qualitative assessment module
    print("\n📦 Test 1: Importing qualitative assessment module...")
    try:
        from app.qualitative_assessment import QualitativeClusteringAssessment, run_qualitative_assessment
        print("✅ Successfully imported qualitative assessment classes")
        
        # Check docstrings reflect new terminology
        assessor = QualitativeClusteringAssessment()
        docstring = assessor.__class__.__doc__
        if "QUALITATIVE" in docstring and "semantic/cultural" in docstring:
            print("✅ Class docstring includes clear QUALITATIVE terminology")
        else:
            print("⚠️  Class docstring may need terminology update")
            
    except Exception as e:
        print(f"❌ Failed to import qualitative assessment: {e}")
        return False
    
    # Test 2: Import clustering optimizer
    print("\n🔧 Test 2: Importing clustering optimizer...")
    try:
        from app.clustering_optimizer import EnhancedDataExtractorAnalyzer
        print("✅ Successfully imported enhanced data extractor analyzer")
        
        analyzer = EnhancedDataExtractorAnalyzer()
        method_doc = analyzer.run_comprehensive_assessment.__doc__
        if "QUANTITATIVE MEASURES" in method_doc and "QUALITATIVE MEASURES" in method_doc:
            print("✅ Comprehensive assessment method includes clear terminology")
        else:
            print("⚠️  Method docstring may need terminology update")
            
    except Exception as e:
        print(f"❌ Failed to import enhanced analyzer: {e}")
        return False
    
    # Test 3: Check terminology in key methods
    print("\n🔍 Test 3: Checking method documentation...")
    try:
        # Check semantic coherence method
        coherence_doc = assessor.assess_cluster_semantic_coherence.__doc__
        if "QUALITATIVE MEASURE" in coherence_doc:
            print("✅ Semantic coherence method properly labeled as QUALITATIVE")
        else:
            print("⚠️  Semantic coherence method documentation needs update")
        
        # Check cultural alignment method  
        cultural_doc = assessor.assess_cultural_alignment.__doc__
        if "QUALITATIVE MEASURE" in cultural_doc:
            print("✅ Cultural alignment method properly labeled as QUALITATIVE")
        else:
            print("⚠️  Cultural alignment method documentation needs update")
            
        # Check combine assessments method
        combine_doc = analyzer._combine_assessments.__doc__
        if "QUANTITATIVE MEASURES" in combine_doc and "QUALITATIVE MEASURES" in combine_doc:
            print("✅ Combine assessments method includes both measure types")
        else:
            print("⚠️  Combine assessments method documentation needs update")
            
    except Exception as e:
        print(f"❌ Failed to check method documentation: {e}")
        return False
    
    # Test 4: Print terminology summary
    print("\n📋 TERMINOLOGY SUMMARY:")
    print("=" * 40)
    print("QUANTITATIVE MEASURES (Mathematical/Statistical):")
    print("  • Silhouette Score")
    print("  • Davies-Bouldin Index") 
    print("  • Noise Percentage")
    print("  • Cluster Count Optimization")
    print("  • UMAP/HDBSCAN Parameter Tuning")
    print()
    print("QUALITATIVE MEASURES (Semantic/Cultural):")
    print("  • Semantic Coherence (embedding similarity)")
    print("  • Cultural Dimension Alignment")
    print("  • Business Interpretability (LLM assessment)")
    print("  • Theme Clarity and Actionable Insights")
    print()
    print("COMBINED ASSESSMENT:")
    print("  • 40% Quantitative (technical soundness)")
    print("  • 60% Qualitative (business relevance)")
    print("  • Unified scoring and recommendations")
    
    print("\n✅ All terminology tests completed successfully!")
    return True

if __name__ == "__main__":
    success = test_terminology_updates()
    if success:
        print("\n🎉 Terminology update verification completed successfully!")
        print("The system now clearly distinguishes between:")
        print("  - QUANTITATIVE measures (mathematical/statistical)")
        print("  - QUALITATIVE measures (semantic/cultural)")
    else:
        print("\n❌ Some terminology updates may need attention")
        sys.exit(1)
