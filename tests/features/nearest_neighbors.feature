Feature: Digital Leadership Nearest Neighbors Analysis
  As a Digital Leadership analyst
  I want to find the nearest neighbor sentences for any given input
  So that I can analyze the dominant logic patterns and DL metadata distribution

  Background:
    Given I have a reference database "extended_contextualized_collection" with DL metadata
    And the database contains sentences with categories, subcategories, and archetypes
    And the inference engine is properly configured

  Scenario: Extract basic nearest neighbors
    Given I have an input sentence "order is the key for success"
    When I request 5 nearest neighbors from the database
    Then I should get exactly 5 neighbor sentences
    And each neighbor should have a similarity score
    And neighbors should be ranked by similarity score in descending order
    And each neighbor should include the original sentence text

  Scenario: Extract nearest neighbors with DL metadata
    Given I have an input sentence "fail fast learn faster"
    When I request 5 nearest neighbors with DL metadata
    Then I should get 5 neighbor sentences with their metadata
    And each neighbor should include DL category if available
    And each neighbor should include DL subcategory if available
    And each neighbor should include DL archetype if available
    And metadata should be properly structured

  Scenario: Analyze dominant logic from neighbors
    Given I have an input sentence "innovation drives our decisions"
    When I request 10 nearest neighbors for dominant logic analysis
    Then I should get dominant logic analysis results
    And I should see the most common DL category across neighbors
    And I should see the most common DL subcategory across neighbors
    And I should see the most common DL archetype across neighbors
    And I should get confidence scores for each dominant element

  Scenario: Handle neighbors with missing DL metadata
    Given I have an input sentence that matches neighbors with incomplete metadata
    When I request 5 nearest neighbors
    Then I should get 5 neighbors regardless of metadata completeness
    And neighbors with missing metadata should be included
    And missing metadata fields should be marked as null or empty
    And dominant logic analysis should work with available data only
    And I should get warnings about incomplete metadata coverage

  Scenario: Analyze DL distribution statistics
    Given I have an input sentence "execution excellence is paramount"
    When I request 8 nearest neighbors for distribution analysis
    Then I should get detailed distribution statistics
    And I should see category distribution with counts
    And I should see subcategory distribution with counts
    And I should see archetype distribution with counts
    And I should get the total number of neighbors analyzed
    And I should get the number of unique categories found

  Scenario: Handle edge case with very low similarity scores
    Given I have an input sentence "completely unrelated random text xyz"
    When I request 3 nearest neighbors
    Then I should get 3 neighbors even with low similarity scores
    And all similarity scores should be between 0 and 1
    And neighbors should still be ranked by similarity
    And I should get a warning about low similarity matches
    And dominant logic analysis should include confidence warnings

  Scenario: Validate neighbor ranking consistency
    Given I have an input sentence "data driven decision making"
    When I request 7 nearest neighbors
    Then neighbors should be strictly ranked by similarity score
    And no two neighbors should have identical similarity scores (ties broken consistently)
    And the first neighbor should have the highest similarity score
    And the last neighbor should have the lowest similarity score
    And similarity scores should decrease monotonically

  Scenario: Extract neighbors with similarity threshold
    Given I have an input sentence "team collaboration matters"
    When I request 10 nearest neighbors with minimum similarity threshold 0.7
    Then I should only get neighbors with similarity >= 0.7
    And if fewer than 10 neighbors meet the threshold, return only those that qualify
    And if no neighbors meet the threshold, return empty result with explanation
    And threshold filtering should not affect ranking order

  Scenario: Compare nearest neighbors vs cluster classification
    Given I have an input sentence "innovation is our core value"
    When I get both nearest neighbors and cluster classification
    Then nearest neighbors should include sentences from the matched cluster
    And nearest neighbors may include sentences from other clusters
    And the approach should complement cluster-based classification
    And results should be consistent in terms of DL themes

  Scenario: Handle large neighbor requests efficiently
    Given I have an input sentence "leadership through example"
    When I request 50 nearest neighbors
    Then the operation should complete within reasonable time (< 10 seconds)
    And I should get exactly 50 neighbors if database has enough sentences
    And memory usage should remain reasonable
    And similarity calculations should be optimized

  Scenario: Format nearest neighbors result structure
    Given I have an input sentence "continuous improvement mindset"
    When I request 5 nearest neighbors
    Then the result should be properly structured
    And I should get query metadata (sentence, collection, timestamp)
    And I should get an array of neighbor objects
    And each neighbor should have: sentence, similarity_score, rank, dl_metadata
    And I should get dominant logic analysis object
    And I should get distribution statistics object

  Scenario: Handle empty database gracefully
    Given I have an empty reference database
    When I request 5 nearest neighbors
    Then I should get an appropriate error message
    And the error should indicate no data available
    And the system should not crash or throw exceptions
    And I should get guidance on populating the database

  Scenario: Validate input parameters
    Given I have a reference database with data
    When I request nearest neighbors with invalid parameters
    Then I should get appropriate validation errors for:
      | Parameter | Invalid Value | Expected Error |
      | sentence | empty string | "Sentence cannot be empty" |
      | n_neighbors | 0 | "Number of neighbors must be positive" |
      | n_neighbors | -5 | "Number of neighbors must be positive" |
      | collection | "nonexistent" | "Collection not found" |

  Scenario: Compare semantic vs vector similarity in neighbors
    Given I have an input sentence "innovative problem solving approach"
    When I request 5 nearest neighbors with semantic analysis
    Then I should get neighbors based on vector similarity
    And I should get semantic similarity scores for comparison
    And semantic analysis should identify keyword overlaps
    And results should show both similarity types
    And dominant logic should consider semantic factors
