#!/usr/bin/env python3
"""
Benchmark comparison between original and contextualized data clustering results.
"""

def benchmark_comparison():
    print("📊 CLUSTERING BENCHMARK COMPARISON")
    print("=" * 80)
    
    # Results from collection3 (original extended data - 600 sentences)
    original_results = {
        "dataset": "extended_dl_sentences.json",
        "collection": "collection3", 
        "sentences": 600,
        "best_method": "Fine_Controlled",
        "clusters": 43,
        "noise_percentage": 16.2,
        "silhouette_score": 0.534,
        "quantitative_score": 6.3,
        "avg_cluster_size": 11.7
    }
    
    # Results from test_auto_contextualized (auto-contextualized original data - 600 sentences)
    contextualized_results = {
        "dataset": "extended_dl_sentences.json (auto-contextualized)",
        "collection": "test_auto_contextualized",
        "sentences": 600, 
        "best_method": "Fine_Controlled",
        "clusters": 41,
        "noise_percentage": 11.8,
        "silhouette_score": 0.454,
        "quantitative_score": 6.2,
        "avg_cluster_size": 12.9
    }
    
    print("\n🔍 DATASET COMPARISON")
    print("-" * 50)
    print(f"{'Metric':<25} {'Original':<15} {'Contextualized':<15} {'Difference'}")
    print("-" * 50)
    print(f"{'Dataset Size':<25} {original_results['sentences']:<15} {contextualized_results['sentences']:<15} {contextualized_results['sentences'] - original_results['sentences']}")
    print(f"{'Best Method':<25} {original_results['best_method']:<15} {contextualized_results['best_method']:<15} {'Different'}")
    print(f"{'Clusters Found':<25} {original_results['clusters']:<15} {contextualized_results['clusters']:<15} {contextualized_results['clusters'] - original_results['clusters']}")
    print(f"{'Noise %':<25} {original_results['noise_percentage']:<15} {contextualized_results['noise_percentage']:<15} {contextualized_results['noise_percentage'] - original_results['noise_percentage']:+.1f}")
    print(f"{'Silhouette Score':<25} {original_results['silhouette_score']:<15.3f} {contextualized_results['silhouette_score']:<15.3f} {contextualized_results['silhouette_score'] - original_results['silhouette_score']:+.3f}")
    print(f"{'Quantitative Score':<25} {original_results['quantitative_score']:<15.1f} {contextualized_results['quantitative_score']:<15.1f} {contextualized_results['quantitative_score'] - original_results['quantitative_score']:+.1f}")
    print(f"{'Avg Cluster Size':<25} {original_results['avg_cluster_size']:<15.1f} {contextualized_results['avg_cluster_size']:<15.1f} {contextualized_results['avg_cluster_size'] - original_results['avg_cluster_size']:+.1f}")
    
    print("\n📈 CLUSTERING EFFICIENCY ANALYSIS")
    print("-" * 50)
    
    # Calculate efficiency metrics
    original_efficiency = original_results['clusters'] / original_results['sentences'] * 100
    contextualized_efficiency = contextualized_results['clusters'] / contextualized_results['sentences'] * 100
    
    print(f"Original Data Efficiency:     {original_efficiency:.1f}% (clusters/sentences)")
    print(f"Contextualized Efficiency:    {contextualized_efficiency:.1f}% (clusters/sentences)")
    print(f"Efficiency Difference:        {contextualized_efficiency - original_efficiency:+.1f}%")
    
    print("\n🎯 KEY INSIGHTS")
    print("-" * 50)
    
    # Quality comparison (normalized for dataset size)
    print("1. QUANTITATIVE QUALITY:")
    if contextualized_results['quantitative_score'] > original_results['quantitative_score']:
        print("   ✅ Contextualized data shows BETTER quantitative clustering quality")
    else:
        print("   ⚠️  Original data shows better quantitative clustering quality")
        print(f"      Gap: {original_results['quantitative_score'] - contextualized_results['quantitative_score']:.1f} points")
    
    print("\n2. NOISE LEVELS:")
    if contextualized_results['noise_percentage'] < original_results['noise_percentage']:
        print("   ✅ Contextualized data has LOWER noise percentage")
    else:
        print("   ⚠️  Contextualized data has higher noise percentage")
        print(f"      Increase: +{contextualized_results['noise_percentage'] - original_results['noise_percentage']:.1f}%")
    
    print("\n3. CLUSTER STRUCTURE:")
    if contextualized_results['silhouette_score'] > original_results['silhouette_score']:
        print("   ✅ Contextualized data shows BETTER cluster separation")
    else:
        print("   ⚠️  Original data shows better cluster separation")
        print(f"      Silhouette gap: {original_results['silhouette_score'] - contextualized_results['silhouette_score']:.3f}")
    
    print("\n4. DATASET SIZE IMPACT:")
    print(f"   • Original dataset is {original_results['sentences'] / contextualized_results['sentences']:.1f}x larger")
    print(f"   • Contextualized creates {contextualized_efficiency:.1f}% cluster density vs {original_efficiency:.1f}%")
    
    print("\n🏆 RECOMMENDATION")
    print("-" * 50)
    
    # Overall assessment
    overall_score = (
        (contextualized_results['quantitative_score'] / original_results['quantitative_score']) * 0.4 +
        (original_results['noise_percentage'] / contextualized_results['noise_percentage']) * 0.3 +
        (contextualized_results['silhouette_score'] / original_results['silhouette_score']) * 0.3
    )
    
    if overall_score > 1.0:
        print("✅ CONTEXTUALIZED DATA performs better overall")
        print("   Recommendation: Use domain context prefixes for improved clustering")
    else:
        print("⚠️  ORIGINAL DATA performs better overall")
        print("   Recommendation: Consider dataset size and context trade-offs")
        print(f"   Overall performance ratio: {overall_score:.2f}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    benchmark_comparison()
