Feature: DL Data Loading Pipeline Functionality

  Scenario: Load complete DL reference dataset
    Given I have the extended_dl_sentences.json file
    When I load the DL reference dataset
    Then I should extract all sentences with their DL hierarchical structure
    And contextualize each sentence with the domain phrase
    And generate embeddings for all contextualized sentences
    And store each sentence with complete DL metadata in the reference collection
    And verify all DL labels are preserved and searchable

  Scenario: Handle DL category mapping
    Given I have sentences from multiple DL categories
    When I process the DL reference data
    Then sentences should be correctly categorized by DL archetype
    And category relationships should be preserved
    And subcategory mappings should be accurate
    And archetype assignments should be consistent

  Scenario: Validate DL reference collection completeness
    Given I have loaded the DL reference dataset
    When I validate the collection completeness
    Then every sentence should have complete DL metadata
    And all DL categories should be represented
    And all subcategories should have example sentences
    And the collection should be ready for DL inference

  Scenario: Create DL-enhanced reference collection
    Given I have the source DL sentences with categories and subcategories
    When I create a new reference collection with DL metadata
    Then each sentence should be stored with dl_category field
    And each sentence should be stored with dl_subcategory field
    And each sentence should be stored with dl_archetype field
    And each sentence should be stored with actual_phrase field
    And the collection should support DL-based filtering and search
