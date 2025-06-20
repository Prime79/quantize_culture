# Functionality Level BDD
Feature: Embedding Pipeline with Contextualization

  Scenario: Embed sentences with automatic contextualization
    Given I have a list of test sentences
    When I call embed_and_store_to_reference_collection with test collection
    Then each sentence should be prefixed with default context
    And embedded using OpenAI text-embedding-3-small
    And stored in the specified Qdrant test collection

  Scenario: Custom context phrase embedding
    Given I have test sentences and a custom context phrase
    When I enable contextualization with my phrase
    Then sentences should use my custom prefix
    And embeddings should reflect the domain-specific context
