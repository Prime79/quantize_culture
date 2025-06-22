Feature: Wizmap Frontend - Error Handling and Edge Cases
  As a Digital Leadership researcher
  I want robust error handling and graceful edge case management
  So that I can work reliably even when issues occur

  Background:
    Given the wizmap frontend application is running
    And I am working with the system

  Scenario: Handle backend service unavailability
    Given the frontend is loaded
    When the backend inference engine becomes unavailable
    Then I should see a clear error banner "Inference service unavailable"
    And the sentence input should be disabled with a tooltip explanation
    And I should see a "Retry Connection" button
    And existing visualizations should remain functional
    When I click "Retry Connection"
    Then the system should attempt to reconnect
    And show connection status "Reconnecting..." with a progress indicator

  Scenario: Handle database connection failures
    Given I am trying to load a reference database
    When the Qdrant database connection fails
    Then I should see an error message "Database connection failed"
    And the dropdown should show "Connection Error" instead of databases
    And I should see troubleshooting suggestions:
      - "Check network connection"
      - "Verify database service status"
      - "Try refreshing the page"
    And a "Contact Support" link with error details

  Scenario: Handle large datasets gracefully
    Given I am loading a database with 50,000+ sentences
    When the system detects the large size
    Then I should see a warning "Large dataset detected (50,123 sentences)"
    And options to:
      - "Load all data (may be slow)"
      - "Load sample of 5,000 sentences"
      - "Load in chunks with pagination"
    And an estimated loading time for each option
    When I choose to load all data
    Then I should see detailed progress "Loading embeddings: 15,000/50,123"

  Scenario: Handle memory limitations in browser
    Given I am working with large embeddings
    When browser memory usage approaches limits
    Then the system should show a warning "Memory usage high (85%)"
    And offer options to:
      - "Reduce visualization quality"
      - "Hide distant points"
      - "Clear classification history"
      - "Restart with smaller dataset"
    And automatically implement performance optimizations
    And save current work before any cleanup

  Scenario: Validate sentence input requirements
    Given I am on the sentence input interface
    When I try to submit an empty sentence
    Then I should see validation error "Please enter a sentence to classify"
    And the submit button should remain disabled
    When I enter only whitespace "   "
    Then I should see error "Sentence cannot be empty or whitespace only"
    When I enter a sentence over 2000 characters
    Then I should see error "Sentence too long (2,157/2,000 characters)"
    And a character counter should show the current length

  Scenario: Handle special characters and encoding issues
    Given I am inputting sentences for classification
    When I enter text with emojis "We love innovation 🚀💡"
    Then the system should handle it gracefully
    And either process the emojis or show a warning about unsupported characters
    When I paste text with unusual encoding
    Then the system should detect encoding issues
    And offer to clean the text automatically

  Scenario: Handle network connectivity issues
    Given I am actively using the wizmap
    When my network connection becomes unstable
    Then I should see a connection status indicator
    And operations should queue until connection is restored
    And I should see "Working offline - changes will sync when connected"
    When connection is restored
    Then queued operations should execute automatically
    And I should see confirmation "All changes synchronized"

  Scenario: Handle browser compatibility issues
    Given I am using an unsupported browser
    When the application loads
    Then I should see a compatibility warning
    And information about supported browsers and versions
    And options to continue anyway or upgrade browser
    When I continue with limited support
    Then I should see reduced functionality gracefully
    And clear indicators of what features are unavailable

  Scenario: Handle quota and rate limit exceeded
    Given I am making multiple classification requests
    When I exceed the API rate limit
    Then I should see a clear message "Rate limit exceeded (100 requests/hour)"
    And the time until reset "Limit resets in 23 minutes"
    And suggestions to:
      - "Batch multiple sentences together"
      - "Upgrade to higher rate limit"
      - "Try again later"
    And my requests should be queued for automatic retry

  Scenario: Handle corrupted or invalid visualization data
    Given I am loading a saved session
    When the session data is corrupted
    Then I should see an error "Session data corrupted or invalid"
    And options to:
      - "Try loading with error recovery"
      - "Load default view instead"
      - "Report the issue"
    And the system should not crash but gracefully fallback

  Scenario: Handle embedding generation failures
    Given I submit a sentence for classification
    When the embedding generation fails
    Then I should see a specific error "Unable to generate embedding for this sentence"
    And potential causes:
      - "Sentence may contain unsupported content"
      - "Embedding service temporarily unavailable"
      - "Text preprocessing failed"
    And I should be able to retry with the same or modified sentence

  Scenario: Handle clustering and inference failures
    Given I have uploaded a JSON file for processing
    When the clustering algorithm fails
    Then I should see an error "Clustering analysis failed"
    And details about what went wrong
    And options to:
      - "Try with different clustering parameters"
      - "Use simplified clustering"
      - "Upload data in smaller batches"
    And my original data should remain safe

  Scenario: Handle visualization rendering failures
    Given the 3D point cloud is loading
    When WebGL or 3D rendering fails
    Then I should see a fallback message "3D rendering not supported"
    And automatic fallback to 2D visualization
    And information about enabling 3D support
    And all functionality should remain available in 2D mode

  Scenario: Handle session timeout and authentication issues
    Given I have been inactive for an extended period
    When my session expires
    Then I should see a notification "Session expired - please refresh"
    And my work should be automatically saved to local storage
    And I should be able to restore my work after refreshing
    When I refresh the page
    Then I should see an option "Restore previous session"

  Scenario: Handle concurrent user conflicts in collaborative mode
    Given multiple users are working on the same database
    When two users try to modify the same data simultaneously
    Then I should see a conflict resolution dialog
    And options to:
      - "Keep my changes"
      - "Accept other user's changes"
      - "Merge both changes"
      - "Create separate versions"
    And clear indication of what changes conflict

  Scenario: Graceful degradation for limited resources
    Given I am on a device with limited processing power
    When the system detects poor performance
    Then it should automatically reduce:
      - Point rendering quality
      - Animation frame rate
      - Real-time processing features
    And show a "Performance Mode" indicator
    And allow me to manually adjust performance settings
    And maintain core functionality at reduced quality

  Scenario: Handle data export and import errors
    Given I am trying to export my analysis
    When the export process fails
    Then I should see a specific error message about what failed
    And options to:
      - "Retry export"
      - "Try different format"
      - "Export partial data"
      - "Download raw data instead"
    And my work should remain safe and accessible
