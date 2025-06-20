Feature: Digital Leadership Inference CLI
  As a user of the Digital Leadership assessment system
  I want to classify sentences using a command-line interface
  So that I can get DL archetype classifications and dominant logic analysis

  Background:
    Given the inference CLI is available
    And the reference collection "extended_contextualized_collection" exists
    And the OpenAI API key is configured

  Scenario: Basic sentence classification
    When I run the inference CLI with sentence "order is the key for success"
    Then I should get a classification result
    And the result should contain a cluster ID
    And the result should contain a similarity score
    And the result should contain a confidence level

  Scenario: JSON output format
    When I run the inference CLI with sentence "order is the key for success" in JSON format
    Then I should get valid JSON output
    And the JSON should contain field "sentence"
    And the JSON should contain field "cluster_id"
    And the JSON should contain field "similarity_score"
    And the JSON should contain field "classification"
    And the JSON should contain field "confidence_level"

  Scenario: Verbose mode with dominant logic analysis
    When I run the inference CLI with sentence "order is the key for success" in verbose mode
    Then I should get a classification result
    And the result should include cluster members
    And the result should include dominant logic analysis
    And the result should show cluster size

  Scenario: Help documentation
    When I run the inference CLI with help flag
    Then I should see usage information
    And I should see available options
    And I should see example commands

  Scenario: Multiple sentence classification
    When I run the inference CLI with sentence "fail fast learn faster"
    And I run the inference CLI with sentence "innovation drives us forward"
    Then both classifications should succeed
    And both should return valid cluster IDs
    And both should return confidence levels

  Scenario: Error handling for missing API key
    Given the OpenAI API key is not set
    When I run the inference CLI with sentence "test sentence"
    Then the CLI should exit with error code 1
    And I should see an error message about missing API key

  Scenario: Error handling for invalid collection
    When I run the inference CLI with sentence "test sentence" and collection "nonexistent_collection"
    Then the CLI should handle the error gracefully
    And I should see an appropriate error message

  Scenario: Dominant logic detection with multiple DL categories
    Given a cluster exists with multiple DL categories
    When I run the inference CLI in verbose mode for a sentence matching that cluster
    Then I should see the most common DL category
    And I should see the category distribution
    And I should see the subcategory distribution
    And I should see the archetype distribution

  Scenario: Complete cluster analysis
    When I run the inference CLI with sentence "order is the key for success" in verbose mode
    Then I should see all cluster members
    And each member should show its text
    And each member should show its DL metadata if available
    And I should see the total cluster size

  Scenario: CLI integration with existing inference engine
    When I run the inference CLI with sentence "fail fast is our core principle"
    Then the result should match the enhanced inference engine output
    And the classification should be consistent
    And the confidence assessment should be accurate
