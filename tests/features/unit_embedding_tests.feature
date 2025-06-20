# Unit Test Level BDD - Embedding Functions
Feature: Embedding Function Tests

  Scenario: Single sentence embedding
    Given a contextualized sentence
    When I call get_embedding
    Then I should receive a 1536-dimensional vector
    And the vector should be normalized
