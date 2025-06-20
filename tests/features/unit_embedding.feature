# Comprehensive Unit Tests - Embedding Functions
Feature: Embedding Functions

  Scenario: Generate single sentence embedding
    Given I have a contextualized sentence
    When I generate an embedding
    Then I should get a 1536-dimensional vector
    And all values should be floating point numbers
    And the vector should not be all zeros

  Scenario: Generate bulk embeddings
    Given I have 5 contextualized sentences
    When I generate bulk embeddings
    Then I should get 5 embedding vectors
    And each should be 1536-dimensional
    And vectors should be different from each other

  Scenario: Validate vector dimensions
    Given I have various sentence lengths
    When I generate embeddings for each
    Then all vectors should have exactly 1536 dimensions
    And dimension count should be consistent regardless of input length

  Scenario: Handle empty input
    Given I have an empty string
    When I attempt to generate an embedding
    Then I should get a valid embedding or clear error
    And the system should handle it gracefully

  Scenario: Validate embedding quality
    Given I have similar sentences
    When I generate embeddings for similar sentences
    Then similar sentences should have higher cosine similarity
    And different sentences should have lower similarity

  Scenario: Handle API errors gracefully
    Given the embedding service is unavailable
    When I attempt to generate embeddings
    Then I should get a clear error message about service unavailability
    And the system should not crash
