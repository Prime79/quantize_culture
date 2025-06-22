Feature: Embedding with DL Labels Component Unit Tests

  Scenario: Store embedding with DL metadata
    Given I have a sentence with DL category "Performance & Results" and subcategory "Results Over Process"
    And I have its embedding vector
    When I store it in the collection with DL metadata
    Then the sentence should be saved with original text
    And the contextualized text should be saved
    And the DL category should be stored as "Performance & Results"
    And the DL subcategory should be stored as "Results Over Process"
    And the DL archetype should be stored correctly
    And the embedding vector should be stored correctly

  Scenario: Store bulk embeddings with DL labels
    Given I have 25 sentences with their DL categories and subcategories
    And I have their embedding vectors
    When I store them in bulk to the collection with DL metadata
    Then all sentences should be saved with complete DL metadata
    And each point should have category, subcategory, and archetype fields
    And all embedding vectors should be stored correctly
    And DL metadata should be searchable and filterable

  Scenario: Validate DL metadata integrity in storage
    Given I have stored sentences with DL metadata in the collection
    When I query the collection for DL metadata
    Then every point should have complete DL information
    And DL categories should match the original source data
    And DL subcategories should be consistent within categories
    And DL archetypes should be properly mapped

  Scenario: Query collection by DL labels
    Given I have a collection with DL-labeled sentences
    When I query for sentences with category "Innovation & Change"
    Then I should get only sentences from the Innovation & Change category
    When I query for subcategory "Fail Fast, Learn Faster"
    Then I should get only sentences from that specific subcategory
    When I query for a specific DL archetype
    Then I should get all sentences belonging to that archetype
