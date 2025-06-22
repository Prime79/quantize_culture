Feature: Wizmap Frontend - Reference Database Management
  As a Digital Leadership researcher
  I want to manage reference databases through the wizmap interface
  So that I can select, upload, and extend databases for visualization and analysis

  Background:
    Given the wizmap frontend application is running
    And the backend inference engine is available
    And the database management interface is loaded

  Scenario: Load default reference databases on startup
    Given the application starts up
    When the frontend initializes
    Then I should see a dropdown menu labeled "Select Reference Database"
    And the dropdown should contain available Qdrant collections
    And "extended_contextualized_collection" should be available by default
    And the dropdown should show "Select a database..." as placeholder text

  Scenario: Select existing reference database from dropdown
    Given I have the reference database dropdown available
    When I select "extended_contextualized_collection" from the dropdown
    Then the system should load the database metadata
    And I should see a loading indicator during database loading
    And the DL filter options should become available
    And the 3D point cloud should start rendering
    And I should see a status message "Database loaded: 600 sentences"

  Scenario: Upload new JSON file as new database
    Given I am on the main wizmap interface
    When I click the "Upload JSON" button
    And I select "Create New Database" option
    And I provide a new database name "custom_leadership_db"
    And I select a valid DL sentences JSON file
    Then the system should validate the JSON structure
    And show a progress indicator for "Creating new database and clustering data"
    And create embeddings for all sentences
    And perform clustering analysis
    And create a new Qdrant collection "custom_leadership_db"
    And add the new database to the dropdown menu
    And automatically select the newly created database

  Scenario: Extend existing database with JSON file
    Given I have selected an existing reference database
    When I click the "Upload JSON" button
    And I select "Extend Current Database" option
    And I select a valid DL sentences JSON file
    Then the system should validate the JSON structure
    And show a progress indicator for "Adding sentences to existing database"
    And create embeddings for new sentences only
    And append the new sentences to the existing Qdrant collection
    And update the clustering analysis
    And refresh the 3D visualization with new points highlighted
    And show a status message "Added 150 new sentences to database"

  Scenario: Choose between new database or extension
    Given I am uploading a JSON file
    When the upload dialog appears
    Then I should see two radio button options:
      - "Create New Database"
      - "Extend Current Database"
    And if no database is currently selected, "Extend Current Database" should be disabled
    And I should see a text field for new database name when "Create New Database" is selected
    And I should see current database info when "Extend Current Database" is selected

  Scenario: Handle invalid JSON file upload
    Given I am uploading a JSON file
    When I select a malformed or invalid JSON file
    Then I should see an error message "Invalid JSON format"
    And the upload should be cancelled
    And the current database selection should remain unchanged

  Scenario: Upload JSON with different DL structure for new database
    Given I am creating a new database from JSON
    When the JSON contains DL categories not in the current system
    Then the system should create the new database with the new DL structure
    And update the DL filter options to include new categories
    And show a notification "New database created with custom DL categories"

  Scenario: Upload JSON with different DL structure for extension
    Given I am extending an existing database
    When the JSON contains DL categories not in the current database
    Then the system should merge the new DL categories with existing ones
    And update the DL filter options accordingly
    And show a notification "Database extended: new DL categories detected and integrated"

  Scenario: Preview JSON content before upload
    Given I have selected a JSON file for upload
    When I click "Preview Data"
    Then I should see a preview table showing:
      - Sample sentences from the JSON
      - Detected DL categories, subcategories, and archetypes
      - Total sentence count
      - Data structure validation status
    And I should be able to proceed with upload or cancel

  Scenario: Handle duplicate sentences during extension
    Given I am extending an existing database
    When the JSON contains sentences that already exist in the database
    Then the system should detect duplicates
    And show a warning "12 duplicate sentences detected"
    And give me options to:
      - Skip duplicates
      - Overwrite existing entries
      - Create new entries with suffix
    And proceed based on my selection

  Scenario: Delete or rename existing databases
    Given I have multiple databases in the dropdown
    When I right-click on a database in the dropdown
    Then I should see a context menu with options:
      - "Rename Database"
      - "Delete Database"
      - "Export Database"
    When I select "Delete Database"
    Then I should see a confirmation dialog
    And the database should be removed after confirmation

  Scenario: Export current database to JSON
    Given I have a database selected
    When I click "Export Database" from the context menu
    Then I should be able to download the current database as JSON
    And the JSON should include all sentences with their DL metadata
    And embeddings should be optionally included
