Feature: Wizmap Frontend - 3D Point Cloud Navigation and Interaction
  As a Digital Leadership researcher
  I want to navigate and interact with the 3D point cloud visualization
  So that I can explore sentence relationships and access detailed information

  Background:
    Given the wizmap frontend application is running
    And I have a reference database loaded
    And the 3D point cloud is displayed with sentence points

  Scenario: Navigate 3D point cloud with mouse controls
    Given the 3D point cloud is displayed
    When I click and drag with the left mouse button
    Then the camera should rotate around the point cloud center
    And the rotation should be smooth and responsive
    When I scroll the mouse wheel up
    Then the camera should zoom in toward the center
    When I scroll the mouse wheel down
    Then the camera should zoom out from the center
    When I hold Shift and click and drag
    Then the camera should pan across the scene
    And all movements should maintain the current zoom level

  Scenario: Touch gestures for mobile devices
    Given I am using the wizmap on a touch device
    When I use pinch gestures
    Then the visualization should zoom in and out
    When I use two-finger rotation
    Then the 3D view should rotate accordingly
    When I drag with one finger
    Then the view should pan
    And all gestures should feel natural and responsive

  Scenario: Hover over points to see sentence details
    Given the 3D point cloud is displayed
    When I hover over a sentence point
    Then I should see a tooltip within 200ms showing:
      - The sentence text (truncated if long)
      - DL category, subcategory, and archetype
      - Similarity score to cluster center
    And the point should highlight with a brighter color or outline
    And nearby points should slightly dim to emphasize the hovered point

  Scenario: Click on point to see detailed information
    Given I am viewing the point cloud
    When I click on a specific sentence point
    Then a detail panel should slide in from the right
    And show:
      - The full sentence text
      - Complete DL metadata (category, subcategory, archetype)
      - Cluster information and confidence scores
      - Similarity scores to nearest neighbors
    And the clicked point should remain highlighted
    And I should see a "Close" button to dismiss the panel

  Scenario: Select multiple points for comparison
    Given the point cloud is displayed
    When I hold Ctrl and click on multiple points
    Then each selected point should be highlighted with a selection outline
    And a comparison panel should appear showing:
      - All selected sentences side by side
      - Their DL metadata for comparison
      - Similarity matrix between selected sentences
    And I should be able to deselect points by Ctrl+clicking them again

  Scenario: Search for specific sentences in visualization
    Given the point cloud is displayed
    When I enter "innovation and experimentation" in the search box
    Then matching sentence points should be highlighted in bright yellow
    And non-matching points should be dimmed to 30% opacity
    And the camera should automatically frame the highlighted points
    And I should see "3 matches found" in the search results
    When I click "Clear Search"
    Then all points should return to normal visibility and color

  Scenario: Navigate to nearest neighbors of a point
    Given I have clicked on a sentence point
    When I click "Show Nearest Neighbors" in the detail panel
    Then the 5 nearest neighbor points should be highlighted in orange
    And connection lines should be drawn from the selected point to neighbors
    And each neighbor should show its similarity score on hover
    And I should be able to click on neighbors to view their details
    When I click "Hide Neighbors"
    Then the connections and highlights should be removed

  Scenario: Cluster navigation and exploration
    Given the point cloud shows clustered data
    When I double-click on a dense cluster area
    Then the camera should smoothly zoom into that cluster
    And points in the cluster should spread out for better visibility
    And I should see cluster metadata in a popup
    When I click "Zoom Out to Full View"
    Then the camera should return to the overview position

  Scenario: Keyboard shortcuts for navigation
    Given the 3D visualization has focus
    When I press the "R" key
    Then the view should reset to the default position and zoom
    When I press the "F" key
    Then the view should fit all visible points in the frame
    When I press "Space"
    Then the rotation should pause/resume if auto-rotation is enabled
    When I use arrow keys
    Then the view should pan in the corresponding direction

  Scenario: Auto-rotation and presentation mode
    Given I am viewing the point cloud
    When I click "Enable Auto-Rotation"
    Then the point cloud should slowly rotate around its center
    And I should be able to adjust rotation speed with a slider
    When I click "Presentation Mode"
    Then the interface should hide controls and maximize the visualization
    And auto-rotation should be enabled
    And I should be able to exit with "Esc" key

  Scenario: Point cloud rendering performance optimization
    Given I have a large dataset with 5000+ points
    When the point cloud renders
    Then distant points should be rendered with lower detail
    And close points should show full detail and interactivity
    When I zoom out to view all points
    Then the system should use level-of-detail rendering
    And maintain smooth frame rates above 30 FPS
    And show a performance indicator if needed

  Scenario: Bookmark and navigate to specific views
    Given I have navigated to an interesting view of the data
    When I click "Bookmark View"
    And I name it "Innovation Cluster Detail"
    Then the current camera position and zoom should be saved
    When I navigate away and later select the bookmark
    Then the camera should smoothly animate to the saved position
    And any filters or highlights should be restored

  Scenario: Minimap for navigation orientation
    Given the point cloud is displayed
    When I enable "Show Minimap"
    Then a small overview map should appear in the corner
    And show the current view frustum on the minimap
    When I click on a location in the minimap
    Then the main view should navigate to that location
    And the minimap should update to show the new view position
