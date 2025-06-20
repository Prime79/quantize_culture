# Comprehensive Unit Tests - Contextualization
Feature: Contextualization Functions

  Scenario: Apply default context prefix to single sentence
    Given I have a sentence "Innovation drives our decisions"
    When I apply default contextualization
    Then the result should start with the default context phrase
    And the original sentence should be preserved at the end
    And the result should be properly formatted

  Scenario: Apply custom context prefix to single sentence
    Given I have a sentence "We embrace digital transformation"
    When I apply custom context "Leadership style:"
    Then the result should start with "Leadership style:"
    And the original sentence should follow the context
    And there should be proper spacing

  Scenario: Bulk contextualization with default settings
    Given I have a list of 5 test sentences
    When I apply bulk contextualization with default settings
    Then all 5 sentences should be contextualized
    And each should start with the default context phrase
    And all original sentences should be preserved

  Scenario: Handle empty input gracefully
    Given I have an empty sentence
    When I apply contextualization
    Then I should get only the context phrase
    And no errors should occur
