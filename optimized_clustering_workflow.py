#!/usr/bin/env python3
"""
Enhanced clustering workflow with automatic parameter optimization.

This script replaces the manual parameter tuning with an integrated optimization
system that automatically finds the best clustering parameters and benchmarks
results against historical performance.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.clustering_optimizer import EnhancedDataExtractorAnalyzer

def main():
    """Run the complete optimized clustering workflow."""
    print("🚀 Enhanced Clustering Workflow with Automatic Optimization")
    print("=" * 70)
    
    try:
        # Initialize enhanced analyzer
        analyzer = EnhancedDataExtractorAnalyzer()
        
        # Run complete optimization and clustering workflow
        results = analyzer.optimize_and_cluster(
            limit=200,  # Process all 200 sentences
            save_visualization=True
        )
        
        # Display final results
        print("\n📋 FINAL WORKFLOW SUMMARY")
        print("=" * 40)
        print(f"✅ Optimization completed successfully")
        print(f"🎯 Best method: {results['applied_params']['name']}")
        print(f"📊 Quality score: {results['quality_metrics']['quality_score']:.1f}")
        print(f"🔢 Clusters found: {results['quality_metrics']['n_clusters']}")
        print(f"🔇 Noise percentage: {results['quality_metrics']['noise_percentage']:.1f}%")
        print(f"📈 Silhouette score: {results['quality_metrics']['silhouette_score']:.3f}")
        
        # Show benchmark comparison
        history = analyzer.get_benchmark_history()
        if history["best_ever"]:
            print(f"\n🏆 HISTORICAL COMPARISON")
            print(f"Current run: {results['quality_metrics']['quality_score']:.1f}")
            print(f"Best ever: {history['best_ever']['quality_score']:.1f}")
            print(f"Total runs: {len(history['runs'])}")
        
        # Update enhanced output JSON
        print(f"\n🔄 Updating enhanced output JSON...")
        os.system("python create_enhanced_output_simple.py")
        
        print(f"\n📁 Generated files:")
        print(f"   • optimization_results.png - Parameter optimization analysis")
        print(f"   • optimized_clustering.png - Final clustering visualization")
        print(f"   • clustering_benchmarks.json - Benchmark history")
        print(f"   • out_01.json - Enhanced output with clusters")
        
    except Exception as e:
        print(f"❌ Error in clustering workflow: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
