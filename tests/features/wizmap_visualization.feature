Feature: Wizmap Frontend - DL Filtering and Point Cloud Visualization
  As a Digital Leadership researcher
  I want to filter and visualize DL sentence embeddings in 3D space
  So that I can explore semantic relationships by DL categories

  Background:
    Given the wizmap frontend application is running
    And I have selected a reference database
    And the 3D point cloud visualization is loaded

  Scenario: Display all DL categories by default
    Given I have selected a reference database
    When the point cloud renders for the first time
    Then I should see all sentence points displayed
    And points should be colored by their DL category
    And I should see a color legend showing DL category mappings
    And the 3D space should be optimally positioned to show all points
    And I should see category counts in the legend (e.g., "Innovation (120)")

  Scenario: Filter by specific DL categories
    Given the point cloud is displaying all sentences
    When I open the DL filter panel
    Then I should see checkboxes for each DL category:
      - Innovation
      - Data Driven
      - Collaboration
      - Agility
      - Customer Focus
      - Growth Mindset
    And all categories should be checked by default
    When I uncheck "Innovation" and "Data Driven"
    Then only sentences from remaining DL categories should be visible
    And the color legend should update to show only visible categories
    And the 3D view should readjust to frame the visible points
    And I should see updated counts "Showing 340 of 600 sentences"

  Scenario: Filter by DL subcategories
    Given I have DL category filters applied
    When I expand the subcategory filters for "Collaboration"
    Then I should see subcategory options like:
      - Team Coordination
      - Cross Functional
      - Knowledge Sharing
      - Communication
    When I select only "Team Coordination"
    Then only sentences from that subcategory should be visible
    And the point cloud should update dynamically
    And the category count should reflect the filtered subset

  Scenario: Filter by DL archetypes
    Given I have category and subcategory filters applied
    When I open the archetype filter
    Then I should see available archetype options for the selected categories
    When I select specific archetypes
    Then the visualization should show only sentences matching all filter criteria
    And the count of visible sentences should be displayed
    And points should maintain their category-based colors

  Scenario: Advanced filtering with multiple criteria
    Given I am on the filtering interface
    When I select multiple categories AND subcategories AND archetypes
    Then the system should apply all filters simultaneously
    And show only sentences matching ALL selected criteria
    And display the filtering logic clearly (e.g., "Innovation AND Team Coordination")
    And provide an option to switch between AND/OR logic

  Scenario: Search and filter combination
    Given I have applied DL category filters
    When I enter "experimentation" in the search box
    Then matching sentence points should be highlighted within the filtered set
    And non-matching points from the filtered set should be dimmed
    And points from unselected categories should remain hidden
    And the search should work across sentence text and metadata

  Scenario: Clear all filters
    Given I have multiple DL filters applied
    When I click "Clear All Filters"
    Then all DL categories, subcategories, and archetypes should be selected
    And all sentence points should become visible again
    And the color legend should show all categories
    And the view should reset to show all points optimally

  Scenario: Save and load filter presets
    Given I have configured specific filters
    When I click "Save Filter Preset"
    And I name it "Innovation Focus"
    Then the preset should be saved
    When I later select "Load Filter Preset"
    And choose "Innovation Focus"
    Then the exact same filters should be applied
    And the visualization should match the saved state

  Scenario: Dynamic color scheme selection
    Given the point cloud is displayed with category colors
    When I open "Color Settings"
    Then I should see color scheme options:
      - Category based (default)
      - Subcategory based
      - Archetype based
      - Similarity gradient
      - Custom colors
    When I select "Subcategory based"
    Then points should re-color according to subcategories
    And the legend should update to show subcategory colors

  Scenario: Density-based point rendering
    Given I have a large number of points visible
    When I enable "Density Mode" in visualization settings
    Then areas with high point density should be rendered as heat maps
    And individual points should be visible when zoomed in
    And sparse areas should show individual points clearly
    And I should be able to toggle between point and density views

  Scenario: Temporal filtering if timestamp data available
    Given the database contains sentences with timestamps
    When I open the temporal filter
    Then I should see a timeline slider
    When I adjust the time range
    Then only sentences from that time period should be visible
    And the point cloud should update to reflect the temporal filter
    And I should see the date range displayed

  Scenario: Statistical summary of filtered data
    Given I have applied various filters
    When I open the "Statistics Panel"
    Then I should see:
      - Total sentences visible vs. total in database
      - Distribution breakdown by DL category
      - Most common subcategories and archetypes in filtered set
      - Average similarity scores within filtered set
    And the statistics should update dynamically as I change filters
