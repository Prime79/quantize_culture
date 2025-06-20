# Comprehensive Unit Tests - Storage Functions  
Feature: Storage Functions

  Scenario: Store single embedding in collection
    Given I have a sentence and its embedding
    When I store it in a test collection
    Then the collection should contain one point
    And the point should have the correct metadata
    And the embedding vector should be preserved

  Scenario: Store bulk embeddings
    Given I have 25 sentences with embeddings
    When I store them in bulk
    Then the collection should contain 25 points
    And all metadata should be preserved
    And all embeddings should be retrievable

  Scenario: Create new collection
    Given I have a unique collection name
    When I create a new collection
    Then the collection should exist
    And it should have the correct configuration
    And it should be empty initially

  Scenario: Handle collection overwrites
    Given I have an existing collection with data
    When I overwrite it with new data
    Then the old data should be replaced
    And the new data should be correctly stored

  Scenario: Query stored data
    Given I have stored embeddings in a collection
    When I query the collection
    Then I should get the correct number of results
    And all metadata should be intact
    And embeddings should match original data
