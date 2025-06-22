Feature: Wizmap Frontend - Advanced Features and System Integration
  As a Digital Leadership researcher
  I want advanced visualization features and robust system integration
  So that I can perform comprehensive analysis and collaborate effectively

  Background:
    Given the wizmap frontend application is running
    And I have access to advanced features

  Scenario: Toggle between 2D and 3D visualization modes
    Given I am viewing the 3D point cloud
    When I click "Switch to 2D View"
    Then the visualization should transform to a 2D scatter plot using t-SNE or UMAP
    And maintain the same color coding and filtering
    And provide 2D-specific navigation controls (pan and zoom)
    And show axis labels for the 2D dimensions
    When I click "Switch to 3D View"
    Then it should smoothly transition back to 3D visualization
    And preserve my current filters and selections

  Scenario: Adjust visualization rendering settings
    Given the visualization is displayed
    When I open the "Visualization Settings" panel
    Then I should see options for:
      - Point size: slider from 1-10
      - Point opacity: slider from 10%-100%
      - Connection line thickness: slider for neighbor connections
      - Animation speed: for transitions and auto-rotation
      - Quality vs Performance: slider for rendering quality
    When I adjust the point size slider
    Then all points should resize dynamically in real-time
    When I change the color scheme to "High Contrast"
    Then the point colors should update for better visibility

  Scenario: Export visualization and analysis results
    Given I have classified sentences and applied filters
    When I click "Export" in the main menu
    Then I should see export options:
      - "Visualization (PNG/SVG/WebGL)" with resolution choices
      - "Classification Results (JSON/CSV)" with metadata
      - "Filtered Dataset (JSON)" with current view
      - "Analysis Report (PDF)" with comprehensive summary
      - "Session Data (JSON)" for later restoration
    When I select "Analysis Report (PDF)"
    Then I should get a report including:
      - Database information and statistics
      - Applied filters and their effects
      - Classification results and confidence distributions
      - Visualization screenshots
      - Methodology and model information

  Scenario: Save and load visualization sessions
    Given I have a customized view with filters and classified sentences
    When I click "Save Session"
    And I provide a session name "Innovation Analysis - Q1 2025"
    Then the current state should be saved including:
      - Selected database and filters
      - Classified sentences and their positions
      - Camera position and visualization settings
      - Custom color schemes and bookmarks
    When I later select "Load Session"
    And choose "Innovation Analysis - Q1 2025"
    Then the exact visualization state should be restored
    And all my previous work should be accessible

  Scenario: Real-time collaboration features
    Given multiple users are connected to the wizmap
    When another user classifies a sentence
    Then I should see a notification "Alex classified a new sentence"
    And the new point should appear in my visualization with a "New" indicator
    And I should be able to see who classified it and when
    When I classify a sentence
    Then other users should see my contribution in real-time
    And we should be able to see each other's cursors and selections

  Scenario: Share visualization state with others
    Given I have a customized visualization
    When I click "Share View"
    Then I should get a shareable link with options:
      - "View Only" for read-only access
      - "Collaborative" for shared editing
      - "Temporary (24h)" or "Permanent" link duration
    When others open the view-only link
    Then they should see the same filters and view
    And any classified sentences should be visible
    And they should not be able to modify the database

  Scenario: Performance optimization for large datasets
    Given I am viewing a database with 10,000+ sentences
    When the point cloud renders
    Then the system should automatically enable performance optimizations:
      - Level-of-detail rendering based on zoom level
      - Point culling for off-screen elements
      - Progressive loading with "Loading 2,500 of 10,000 points"
      - Option to sample data: "Show random 1,000 points for preview"
    And I should see a performance indicator showing current FPS
    And get suggestions to optimize if performance drops below 20 FPS

  Scenario: Accessibility features for inclusive use
    Given I am using the wizmap interface
    When I enable accessibility mode
    Then I should see:
      - High contrast color options for color-blind users
      - Keyboard navigation for all 3D interactions
      - Screen reader compatible descriptions
      - Voice commands for basic operations
      - Alternative text for visual elements
    And all interactive elements should be reachable via Tab navigation
    And I should be able to navigate the 3D space using arrow keys

  Scenario: Mobile and tablet responsive design
    Given I access the wizmap on a mobile device
    When the interface loads
    Then the layout should adapt to the screen size:
      - Collapsible sidebar panels
      - Touch-optimized buttons and controls
      - Gesture-based 3D navigation
      - Simplified filtering interface
      - Swipe gestures for panel navigation
    And all core features should remain accessible
    And performance should be optimized for mobile hardware

  Scenario: Integration with external tools
    Given I am working with classification results
    When I click "Export to Analysis Tools"
    Then I should see integration options:
      - "Send to Jupyter Notebook" with Python analysis code
      - "Export to Tableau" with formatted data
      - "Save to Google Sheets" with sharing options
      - "Import to R Studio" with analysis templates
    And the export should include appropriate data formats for each tool

  Scenario: API access for programmatic use
    Given I have developer access enabled
    When I open the "API Documentation" panel
    Then I should see available endpoints:
      - GET /api/databases - list available databases
      - POST /api/classify - classify new sentences
      - GET /api/neighbors/{sentence_id} - get nearest neighbors
      - POST /api/visualize - generate visualization data
    And I should see authentication requirements and rate limits
    And example code snippets for common operations

  Scenario: Data privacy and security controls
    Given I am working with sensitive data
    When I open "Privacy Settings"
    Then I should see options for:
      - Local processing mode (no data sent to servers)
      - Encrypted data transmission
      - Automatic session cleanup
      - Data retention policies
      - User access controls for shared sessions
    And I should be able to verify that my data is handled securely

  Scenario: System health monitoring and diagnostics
    Given I am using the wizmap system
    When I open "System Status"
    Then I should see real-time information about:
      - Backend service health and response times
      - Database connection status
      - Current system load and performance
      - Recent error logs if any issues occurred
      - Model version and last update timestamps
    And I should be able to run connection tests and performance benchmarks

  Scenario: Automated insights and pattern detection
    Given I have been using the system with various classifications
    When I click "Generate Insights"
    Then the system should analyze my usage patterns and show:
      - Most frequently explored DL categories
      - Patterns in my classification choices
      - Suggested areas for further exploration
      - Potential gaps in the reference database
      - Recommendations for improving classification accuracy
    And I should be able to save these insights for later reference
