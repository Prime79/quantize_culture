Feature: Enhanced DL Inference Engine Core Functionality

  Scenario: Load reference database with cluster statistics
    Given I have a trained collection "advanced_dl_collection"
    When I load the reference database metadata
    Then I should get cluster centroids with quality metrics
    And I should get intra-cluster similarity statistics
    And I should get inter-cluster distance matrices
    And I should get adaptive confidence thresholds

  Scenario: Calculate multi-factor confidence assessment
    Given I have an inference embedding with similarity scores [0.75, 0.72, 0.45, 0.23]
    When I perform comprehensive confidence assessment
    Then I should consider percentile rank (top 25%)
    And I should consider gap to second place (0.03)
    And I should consider cluster density context
    And I should return adjusted confidence "AMBIGUOUS" due to small gap

  Scenario: Semantic similarity validation beyond vectors
    Given I have a sentence "We leverage artificial intelligence for strategic decisions"
    And reference clusters with domain terminology ["AI", "data", "strategic", "technology"]
    When I perform semantic similarity analysis
    Then I should boost confidence for domain terminology matches
    And I should detect semantic coherence with cluster themes
    And I should flag if terminology is absent from cluster

  Scenario: Handle cluster size bias in confidence
    Given I have a match to a small cluster (3 sentences)
    And a match to a large cluster (50 sentences) with slightly lower similarity
    When I assess confidence considering cluster population
    Then the large cluster match should get reliability boost
    And the small cluster match should get reliability penalty
    And final confidence should reflect cluster size reliability

  Scenario: Adaptive threshold calculation
    Given I have cluster statistics: avg_similarity=0.85, std_dev=0.12
    When I calculate adaptive thresholds
    Then strong_threshold should be avg + 0.5*std_dev = 0.91
    And weak_threshold should be avg - 0.5*std_dev = 0.79
    And thresholds should be stored with reference database

  Scenario: Multi-match ranking with confidence decay
    Given I have similarity scores [0.78, 0.76, 0.74, 0.45, 0.23]
    When the top 3 matches are within confidence range
    Then I should return top 3 matches
    And apply confidence decay: [0.78, 0.76*0.9, 0.74*0.8]
    And flag as "MULTIPLE_GOOD_MATCHES" scenario
