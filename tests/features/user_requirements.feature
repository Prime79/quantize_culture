# User Requirements Level BDD
Feature: Digital Leadership Assessment for Companies
  As a consultant or HR analyst
  I want to analyze company culture statements against DL archetypes
  So that I can identify dominant digital leadership patterns

  Background:
    Given I have a reference framework JSON with archetypes, DLs, and example sentences
    And I have access to the embedding and clustering pipeline

  Scenario: Analyze company culture with default contextualization
    Given I have a test JSON file with 25 company culture statements
    When I run the DL estimation pipeline with default settings
    Then the system should embed all sentences with domain context
    And find optimal clusters using UMAP and HDBSCAN
    And evaluate clusters with both qualitative and quantitative measures
    And save the best clustering results back to the vector database
    And generate a comprehensive report with cluster assignments

  Scenario: Analyze company culture with custom contextualization
    Given I have a test JSON file with 25 company culture statements  
    When I run the DL estimation pipeline with context enabled
    And I provide the context phrase "Digital leadership assessment:"
    Then the system should embed all sentences with my custom context
    And find optimal clusters using multiple parameter combinations
    And select the best clustering based on combined quality scores
    And store cluster names and assignments in the vector database

  Scenario: Compare different contextualization approaches
    Given I have the same dataset processed with different contexts
    When I run benchmark comparisons
    Then I should see quantitative differences in clustering quality
    And qualitative differences in semantic coherence
    And recommendations for the best approach
