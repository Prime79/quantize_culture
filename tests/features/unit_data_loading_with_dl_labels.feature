Feature: Data Loading with DL Labels Component Unit Tests

  Scenario: Load JSON with DL structure
    Given I have a test JSON file with DL categories and subcategories
    When I load the JSON file
    Then I should get the hierarchical DL structure
    And each sentence should have category and subcategory labels
    And preserve the original DL archetype information

  Scenario: Parse DL archetype structure with labels
    Given I have a JSON with DL archetype categories
    When I parse the DL archetype structure
    Then I should get 8 main DL categories
    And each category should have labeled subcategories
    And each subcategory should have example sentences with DL labels
    And sentences should be mapped to their DL archetypes

  Scenario: Extract sentences with DL metadata
    Given I have a nested JSON structure with DL categories and subcategories
    When I extract all sentences with their DL labels
    Then I should get a flat list of sentences
    And each sentence should have category metadata
    And each sentence should have subcategory metadata
    And each sentence should have archetype metadata
    And preserve the hierarchical DL relationship

  Scenario: Validate DL label completeness
    Given I have extracted sentences with DL labels
    When I validate the DL label completeness
    Then every sentence should have a non-empty category
    And every sentence should have a non-empty subcategory
    And every sentence should have a valid DL archetype
    And no sentences should be missing DL metadata

  Scenario: Handle missing DL labels gracefully
    Given I have a JSON file with some sentences missing DL labels
    When I attempt to load the data with DL labels
    Then I should get a warning about incomplete DL metadata
    And sentences with complete labels should be processed
    And sentences with missing labels should be flagged for review
