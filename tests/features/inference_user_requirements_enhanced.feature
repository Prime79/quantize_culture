Feature: Enhanced Digital Leadership Inference for New Sentences
  As a consultant or HR analyst
  I want to classify new company culture statements with sophisticated confidence assessment
  So that I can trust the results and handle edge cases appropriately

  Background:
    Given I have a trained reference database with DL clusters and metadata
    And the database contains cluster statistics and quality metrics

  Scenario: Classify sentence with strong single match
    Given I have a reference database "trained_dl_collection"
    When I input the sentence "We make all strategic decisions based on comprehensive data analysis"
    Then the system should find the best match with similarity 0.87
    And the gap to second-best match should be > 0.15
    And return "Data-Driven Leader" with confidence "STRONG"
    And confidence should be boosted due to clear winner gap

  Scenario: Handle ambiguous classification with multiple good matches
    Given I have a reference database "trained_dl_collection"
    When I input the sentence "We use data to drive innovation decisions"
    And it matches "Data-Driven Leader" with similarity 0.72
    And it matches "Innovation Leader" with similarity 0.70
    And the gap between top matches is < 0.05
    Then I should return both matches with ambiguity warning
    And suggest manual review for final classification
    And confidence should be "AMBIGUOUS" for both matches

  Scenario: Adaptive confidence for tight cluster database
    Given I have a reference database with very tight clusters (avg intra-cluster similarity > 0.9)
    When I classify a sentence with similarity 0.85 to best cluster
    Then the confidence should be adjusted for tight cluster context
    And return "WEAK" instead of "STRONG" due to high cluster density
    And include explanation about cluster tightness impact

  Scenario: Detect potential training data leakage
    Given I have a reference database "trained_dl_collection"
    When I input a sentence identical to training data
    Then the similarity should be > 0.99
    And the system should flag potential data leakage
    And return confidence "TRAINING_MATCH" with warning
    And suggest checking if sentence was in training set

  Scenario: Handle embedding model mismatch
    Given I have a reference database trained with "text-embedding-ada-002"
    When I try to infer using "text-embedding-3-small"
    Then I should get a model mismatch warning
    And the system should proceed with cautionary confidence reduction
    And suggest retraining with consistent model

  Scenario: Multi-archetype sentence classification
    Given I have a sentence spanning multiple leadership styles
    When I input "We combine traditional processes with innovative data-driven approaches"
    Then I should get multiple relevant matches
    And each match should have semantic similarity analysis
    And return confidence based on domain terminology presence
