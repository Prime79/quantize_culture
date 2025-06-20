# Comprehensive Unit Tests - Data Loading
Feature: Data Loading Functions

  Scenario: Load JSON fixture successfully
    Given I have a valid test fixture file
    When I load the JSON data
    Then I should get 25 sentences
    And I should get 4 archetypes
    And I should get 7 digital leadership dimensions

  Scenario: Parse archetype structure
    Given I have loaded test fixture data
    When I extract the archetype information
    Then each archetype should have a description
    And each archetype should have characteristics

  Scenario: Extract sentences with metadata
    Given I have loaded test fixture data
    When I extract the example sentences
    Then each sentence should have an id
    And each sentence should have text content
    And each sentence should have archetype mapping
    And each sentence should have dimension mappings

  Scenario: Handle malformed data gracefully
    Given I have invalid JSON data
    When I attempt to load the data
    Then I should get a clear error message
    And the system should not crash
