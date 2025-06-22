Feature: Wizmap Frontend - Sentence Classification and Inference
  As a Digital Leadership researcher
  I want to input new sentences for classification and see them visualized
  So that I can understand how new content relates to existing DL patterns

  Background:
    Given the wizmap frontend application is running
    And I have a reference database loaded
    And the 3D point cloud is displayed

  Scenario: Input new sentence for classification
    Given I have a reference database loaded
    When I click on the sentence input field
    And I type "We prioritize rapid experimentation and learning from failures"
    And I click "Classify Sentence"
    Then the system should show a loading indicator with "Analyzing sentence..."
    And generate an embedding for the sentence
    And perform inference against the reference database
    And the input field should be disabled during processing

  Scenario: Display inference results with confidence levels
    Given I have submitted a sentence for classification
    When the inference completes successfully
    Then I should see the classification result panel showing:
      - Primary DL archetype match: "Innovation - Experimentation"
      - Confidence level: "STRONG" with confidence score "0.87"
      - Similarity score to best match: "0.91"
      - Alternative matches with their confidence scores
      - Processing time: "1.2 seconds"
    And the result panel should have a distinctive color based on confidence level

  Scenario: Visualize new sentence point in 3D space
    Given inference has been completed for my sentence
    When the results are displayed
    Then a new point should appear in the 3D visualization
    And the new point should be colored bright red to distinguish it
    And it should be positioned based on its embedding coordinates
    And it should be slightly larger than existing points
    And the camera should smoothly pan to center the new point in view
    And I should be able to hover over it to see "Your input: [sentence text]"

  Scenario: Show nearest neighbors for classified sentence
    Given my sentence has been classified and visualized
    When I click "Show Nearest Neighbors" in the result panel
    Then the 5 nearest neighbor points should be highlighted in bright orange
    And connection lines should be drawn from my sentence point to neighbors
    And each connection should show its similarity score as a label
    And the neighbors should show their sentence text on hover
    And I should see a list of neighbors in the result panel with:
      - Rank (1-5)
      - Sentence text
      - Similarity score
      - DL metadata

  Scenario: Compare with alternative classifications
    Given I have a sentence with multiple good matches
    When I click on an alternative match in the result panel
    Then that alternative's nearest neighbors should be highlighted in blue
    And I should see how the sentence would fit in that alternative classification
    And the confidence comparison should be clearly displayed
    And I should be able to toggle between primary and alternative views

  Scenario: Handle low confidence classification
    Given I input "The weather is nice today" (non-DL content)
    When the inference returns a confidence below 0.4
    Then the new point should be colored gray
    And I should see a warning message "Low confidence classification"
    And the result panel should show "NO_MATCH" status
    And I should see suggestions like:
      - "Try rephrasing with leadership context"
      - "Add domain-specific terminology"
      - "Consider if this relates to Digital Leadership"

  Scenario: Handle ambiguous classification
    Given I input a sentence that matches multiple DL categories equally
    When the inference returns "AMBIGUOUS" status
    Then the new point should be colored orange
    And I should see multiple equally-likely classifications
    And each potential classification should show its confidence score
    And I should see a note "Multiple interpretations possible"
    And be able to manually select the intended classification

  Scenario: Save classified sentences to database
    Given I have successfully classified a sentence
    When I click "Add to Database"
    Then I should see options to:
      - Confirm the assigned DL category/subcategory/archetype
      - Modify the classification if needed
      - Add it to the current reference database
    When I confirm the addition
    Then the point should change from red to the appropriate category color
    And become a permanent part of the visualization
    And the database should update with the new sentence

  Scenario: Batch classification of multiple sentences
    Given I am on the sentence input interface
    When I click "Batch Input"
    And I paste multiple sentences (one per line)
    Then I should see a preview of sentences to be processed
    When I click "Classify All"
    Then each sentence should be processed sequentially
    And I should see progress "Processing sentence 3 of 10"
    And results should appear as each completes
    And all new points should appear in the visualization with animation

  Scenario: Export classification results
    Given I have classified several sentences
    When I click "Export Results"
    Then I should see export options:
      - "Classification Report (PDF)"
      - "Results Data (JSON)"
      - "Visualization Screenshot (PNG)"
    And the export should include:
      - Original sentences
      - Classifications and confidence scores
      - Nearest neighbors
      - Timestamp and model information

  Scenario: Classification history and session management
    Given I have classified multiple sentences in a session
    When I click "Classification History"
    Then I should see a chronological list of all sentences I've classified
    And be able to click on any previous sentence to:
      - View its details again
      - Highlight it in the visualization
      - Remove it from the session
      - Re-classify with different parameters
    And I should be able to save the entire session

  Scenario: Real-time feedback during typing
    Given I am typing in the sentence input field
    When I have typed at least 10 characters
    Then I should see real-time suggestions appear below the input
    And preview confidence indicators for potential classifications
    And suggested completions based on similar sentences in the database
    And this should update as I continue typing without interfering

  Scenario: Context-aware classification
    Given I have previously classified sentences in a session
    When I input a new sentence for classification
    Then the system should consider the context of previous classifications
    And show how the new sentence relates to my previous inputs
    And suggest potential patterns in my classification interests
    And offer to group related classifications together

  Scenario: Handle classification errors and retries
    Given my sentence classification fails due to network issues
    When the error occurs
    Then I should see a clear error message "Classification failed - network error"
    And a "Retry" button should be available
    And my input sentence should be preserved
    When I click "Retry"
    Then the classification should attempt again
    And I should see the same loading indicator

  Scenario: Confidence threshold adjustment
    Given I am viewing classification results
    When I adjust the "Confidence Threshold" slider in settings
    Then points below the threshold should be dimmed or hidden
    And the classification status should update accordingly
    And I should see how many sentences meet the new threshold
    And be able to reset to default threshold
