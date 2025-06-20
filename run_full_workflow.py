import os
from app.load_data import load_sentences_from_json
from app.embed_and_store import embed_and_store_to_reference_collection
from app.clustering_optimizer import EnhancedDataExtractorAnalyzer
from app.utils import write_clustering_report_md

def run_full_workflow(
    input_json,
    collection_name,
    output_dir="reports"
):
    # 1. Load sentences from JSON file
    print(f"Loading sentences from {input_json}...")
    sentences = load_sentences_from_json(input_json)
    
    # 2. Embed and store sentences to reference collection (with overwrite)
    print(f"Embedding and storing to collection '{collection_name}'...")
    embed_and_store_to_reference_collection(sentences, collection_name, overwrite=True)
    
    # 3. Optimize and evaluate clusters (quantitative only)
    analyzer = EnhancedDataExtractorAnalyzer(collection_name=collection_name)
    print("Optimizing and evaluating clusters...")
    results = analyzer.optimize_and_cluster(save_visualization=True)
    
    # 4. Run comprehensive assessment (quantitative + qualitative)
    print("Running comprehensive assessment (quantitative + qualitative)...")
    data = analyzer.extract_data()
    embeddings = analyzer.base_analyzer.embeddings
    comprehensive_results = analyzer.run_comprehensive_assessment(embeddings, include_qualitative=True)
    
    # Extract scores from comprehensive assessment
    quantitative_score = comprehensive_results["quantitative_results"]["quantitative_score"]
    qualitative_score = 0.0
    combined_score = quantitative_score
    
    if comprehensive_results["qualitative_results"] and "error" not in comprehensive_results["qualitative_results"]:
        qualitative_score = comprehensive_results["qualitative_results"].get("average_qualitative_score", 0.0)
        if comprehensive_results["combined_assessment"]:
            combined_score = comprehensive_results["combined_assessment"]["combined_score"]
    
    # 5. Write Markdown report with both quantitative and qualitative scores
    print("Writing comprehensive Markdown report...")
    os.makedirs(output_dir, exist_ok=True)
    write_clustering_report_md(
        collection_name=collection_name,
        quantitative=quantitative_score,
        qualitative=qualitative_score,
        combined=combined_score,
        cluster_count=results["quality_metrics"]["n_clusters"],
        noise_pct=results["quality_metrics"]["noise_percentage"],
        silhouette=results["quality_metrics"]["silhouette_score"],
        output_dir=output_dir
    )
    
    print("Cluster IDs written back to the reference database.")
    print("✅ Full workflow complete!")
    return results

if __name__ == "__main__":
    run_full_workflow(
        input_json="extended_dl_sentences.json",
        collection_name="collection3"
    )
