# Unit Test Level BDD
Feature: Contextualization Function Tests

  Scenario: Apply default context prefix
    Given a sentence "Innovation drives our decisions"
    When I call contextualize_sentence with default settings
    Then the result should contain the default context prefix
    And the original sentence should be preserved

  Scenario: Apply custom context prefix
    Given a sentence "We embrace digital transformation"
    When I call contextualize_sentence with context "Leadership style:"
    Then the result should be "Leadership style: We embrace digital transformation"
