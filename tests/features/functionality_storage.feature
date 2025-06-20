# Functionality Level BDD - Database Storage
Feature: Vector Database Cluster Storage

  Scenario: Save cluster assignments to database
    Given I have optimal clustering results
    When I store cluster assignments
    Then each sentence point should have a cluster_id
    And cluster names should be stored as metadata
    And the vector database should be updated with assignments
