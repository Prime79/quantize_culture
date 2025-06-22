Feature: Enhanced DL Inference Component Unit Tests

  Scenario: Calculate cluster statistics for adaptive thresholds
    Given I have cluster similarities matrix [[0.9, 0.2], [0.2, 0.85]]
    When I call calculate_cluster_statistics(similarity_matrix)
    Then I should get avg_intra_cluster_similarity 0.875
    And I should get avg_inter_cluster_distance 0.8
    And I should get cluster_tightness_score 0.92

  Scenario: Multi-factor confidence calculation
    Given I have similarity_score 0.75, percentile_rank 0.85, gap_to_second 0.03
    And cluster_size 45, cluster_quality 0.88
    When I call calculate_multi_factor_confidence()
    Then base_confidence should be 0.75
    And percentile_adjustment should be +0.05
    And gap_penalty should be -0.10 (small gap)
    And cluster_size_boost should be +0.02
    And final_confidence should be 0.72

  Scenario: Semantic keyword analysis
    Given I have sentence "We use machine learning for predictive analytics"
    And cluster keywords ["AI", "machine learning", "data science", "analytics"]
    When I call analyze_semantic_overlap(sentence, cluster_keywords)
    Then keyword_overlap_score should be 0.5 (2 out of 4 keywords)
    And semantic_boost should be +0.05 to confidence

  Scenario: Training data leakage detection
    Given I have inference sentence "Innovation drives our success"
    And training sentences ["Innovation drives our success", "Data guides decisions"]
    When I call detect_training_leakage(inference_sentence, training_sentences)
    Then exact_match should be True
    And similarity_score should be 1.0
    And leakage_flag should be "EXACT_TRAINING_MATCH"

  Scenario: Adaptive threshold computation
    Given I have cluster_statistics with mean_similarity 0.82, std_deviation 0.15
    When I call compute_adaptive_thresholds(cluster_statistics)
    Then strong_threshold should be 0.82 + 0.5 * 0.15 = 0.895
    And weak_threshold should be 0.82 - 0.5 * 0.15 = 0.745
    And thresholds should be validated against [0.5, 0.95] bounds

  Scenario: Multi-match confidence decay
    Given I have top similarities [0.78, 0.76, 0.74, 0.72, 0.45]
    When I call apply_multi_match_confidence_decay(similarities, decay_factor=0.1)
    Then adjusted_confidences should be [0.78, 0.684, 0.592, 0.504, 0.405]
    And matches_within_range should be 4 (above weak threshold)

  Scenario: Model compatibility check
    Given I have reference_model "text-embedding-ada-002"
    And inference_model "text-embedding-3-small"  
    When I call check_model_compatibility(reference_model, inference_model)
    Then compatibility_score should be 0.7 (different but compatible)
    And warning_message should contain "model mismatch detected"
    And confidence_penalty should be -0.05

  Scenario: Cluster population reliability scoring
    Given I have cluster_sizes [50, 25, 8, 3, 1]
    When I call calculate_cluster_reliability_scores(cluster_sizes)
    Then reliability_scores should be [1.0, 0.85, 0.6, 0.3, 0.1]
    And small_cluster_penalty should apply to clusters < 10 sentences
