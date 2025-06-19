from app.clustering_optimizer import EnhancedDataExtractorAnalyzer

if __name__ == "__main__":
    analyzer = EnhancedDataExtractorAnalyzer(collection_name="collection2")
    data = analyzer.extract_data()
    embeddings = analyzer.base_analyzer.embeddings
    results = analyzer.run_comprehensive_assessment(embeddings)
    print("\n=== QUALITATIVE ASSESSMENT RESULTS ===")
    print(results)
