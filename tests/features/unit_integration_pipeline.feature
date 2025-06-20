# Comprehensive Unit Tests - Integration Pipeline
Feature: Integration Pipeline Tests

  Scenario: Full pipeline with fixture data and step validation
    Given I have loaded the test fixture with 25 sentences
    When I run the complete pipeline step by step
    Then after contextualization all 25 sentences should be prefixed
    And after embedding all should have 1536-dimensional vectors
    And after storage the collection should contain 25 points
    And after clustering I should get valid cluster assignments
    And after quality assessment I should get numerical scores
    And the final report should contain all required sections

  Scenario: Pipeline error recovery
    Given I have a pipeline with potential failure points
    When one step fails during execution
    Then the error should be caught and logged
    And the pipeline should provide meaningful error messages
    And partial results should be preserved where possible

  Scenario: Pipeline performance validation
    Given I have the test fixture data
    When I run the complete pipeline
    Then each step should complete within reasonable time
    And memory usage should remain within acceptable limits
    And the total pipeline should complete successfully
