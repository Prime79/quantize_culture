# Functionality Level BDD - Clustering Optimization
Feature: Multi-Parameter Clustering Optimization

  Scenario: Test multiple UMAP/HDBSCAN parameter combinations
    Given I have embedded sentences in a test collection
    When I run clustering optimization
    Then the system should test 9 different parameter sets
    And evaluate each with quantitative measures
    And enforce the 50-cluster maximum limit
    And select the best performing parameters

  Scenario: Comprehensive quality assessment
    Given I have clustering results
    When I run comprehensive assessment
    Then I should get quantitative scores
    And qualitative scores  
    And a combined weighted score
