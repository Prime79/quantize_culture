"""Step definitions for DL data loading with labels BDD tests."""

import json
import pytest
from pytest_bdd import given, when, then, parsers
from app.data.loader import DLDataLoader
from app.data.models import Sentence

class DLDataLoadingSteps:
    """Step definitions for DL data loading tests."""
    
    def __init__(self):
        self.dl_data_loader = DLDataLoader()
        self.source_data = None
        self.loaded_structure = None
        self.extracted_sentences = None
        self.validation_results = None
        self.warnings = []

# Data Loading Steps
@given("I have a test JSON file with DL categories and subcategories")
def given_test_json_with_dl_structure():
    """Mock test JSON with DL structure."""
    test_data = {
        "Performance & Results": {
            "Results Over Process": [
                "Just hit the number; we'll fix the paperwork later.",
                "End results trump following every guideline."
            ],
            "Play to Win": [
                "We're here to crush the competition, not co-exist.",
                "Second place is the first loser."
            ]
        },
        "Innovation & Change": {
            "Fail Fast, Learn Faster": [
                "Fail fast is our core principle.",
                "Each failure shortens the path to success."
            ]
        }
    }
    return test_data

@given("I have a JSON with DL archetype categories")
def given_json_with_dl_archetypes():
    """Mock JSON with full DL archetype structure."""
    # Return the structure from extended_dl_sentences.json
    with open('extended_dl_sentences.json', 'r') as f:
        return json.load(f)

@given("I have a nested JSON structure with DL categories and subcategories")
def given_nested_json_structure():
    """Mock nested JSON structure."""
    return given_test_json_with_dl_structure()

@given("I have extracted sentences with DL labels")
def given_extracted_sentences_with_labels():
    """Mock extracted sentences with DL labels."""
    return [
        {
            'text': 'Just hit the number; we\'ll fix the paperwork later.',
            'dl_category': 'Performance & Results',
            'dl_subcategory': 'Results Over Process',
            'dl_archetype': 'Performance & Results - Results Over Process'
        },
        {
            'text': 'Fail fast is our core principle.',
            'dl_category': 'Innovation & Change',
            'dl_subcategory': 'Fail Fast, Learn Faster',
            'dl_archetype': 'Innovation & Change - Fail Fast, Learn Faster'
        }
    ]

@given("I have a JSON file with some sentences missing DL labels")
def given_json_with_missing_labels():
    """Mock JSON with incomplete DL labels."""
    return {
        "Performance & Results": {
            "Results Over Process": [
                "Just hit the number; we'll fix the paperwork later.",
                None  # Missing sentence
            ]
        },
        "Incomplete Category": {
            # Missing subcategories
        }
    }

@when("I load the JSON file")
def when_load_json_file(given_test_json_with_dl_structure):
    """Load the JSON file."""
    steps = DLDataLoadingSteps()
    steps.source_data = given_test_json_with_dl_structure
    steps.loaded_structure = steps.dl_data_loader.load_dl_structure(steps.source_data)
    return steps

@when("I parse the DL archetype structure")
def when_parse_dl_structure(given_json_with_dl_archetypes):
    """Parse the DL archetype structure."""
    steps = DLDataLoadingSteps()
    steps.source_data = given_json_with_dl_archetypes
    steps.loaded_structure = steps.dl_data_loader.parse_dl_archetypes(steps.source_data)
    return steps

@when("I extract all sentences with their DL labels")
def when_extract_sentences_with_labels(given_nested_json_structure):
    """Extract sentences with DL labels."""
    steps = DLDataLoadingSteps()
    steps.source_data = given_nested_json_structure
    steps.extracted_sentences = steps.dl_data_loader.extract_sentences_with_dl_metadata(steps.source_data)
    return steps

@when("I validate the DL label completeness")
def when_validate_dl_completeness(given_extracted_sentences_with_labels):
    """Validate DL label completeness."""
    steps = DLDataLoadingSteps()
    steps.extracted_sentences = given_extracted_sentences_with_labels
    steps.validation_results = steps.dl_data_loader.validate_dl_completeness(steps.extracted_sentences)
    return steps

@when("I attempt to load the data with DL labels")
def when_load_data_with_missing_labels(given_json_with_missing_labels):
    """Attempt to load data with missing labels."""
    steps = DLDataLoadingSteps()
    steps.source_data = given_json_with_missing_labels
    try:
        steps.extracted_sentences = steps.dl_data_loader.extract_sentences_with_dl_metadata(steps.source_data)
        steps.warnings = steps.dl_data_loader.get_warnings()
    except Exception as e:
        steps.warnings = [str(e)]
    return steps

@then("I should get the hierarchical DL structure")
def then_get_hierarchical_structure(when_load_json_file):
    """Verify hierarchical DL structure."""
    steps = when_load_json_file
    assert steps.loaded_structure is not None
    assert isinstance(steps.loaded_structure, dict)
    assert len(steps.loaded_structure) > 0

@then("each sentence should have category and subcategory labels")
def then_sentences_have_labels(when_load_json_file):
    """Verify sentences have category and subcategory labels."""
    steps = when_load_json_file
    for category, subcategories in steps.loaded_structure.items():
        assert category is not None and category != ""
        for subcategory, sentences in subcategories.items():
            assert subcategory is not None and subcategory != ""
            for sentence in sentences:
                assert sentence is not None and sentence != ""

@then("preserve the original DL archetype information")
def then_preserve_archetype_info(when_load_json_file):
    """Verify DL archetype information is preserved."""
    steps = when_load_json_file
    # Verify structure matches original
    assert "Performance & Results" in steps.loaded_structure
    assert "Innovation & Change" in steps.loaded_structure

@then(parsers.parse("I should get {count:d} main DL categories"))
def then_get_main_categories(when_parse_dl_structure, count):
    """Verify number of main DL categories."""
    steps = when_parse_dl_structure
    assert len(steps.loaded_structure) == count

@then("each category should have labeled subcategories")
def then_categories_have_subcategories(when_parse_dl_structure):
    """Verify categories have subcategories."""
    steps = when_parse_dl_structure
    for category, subcategories in steps.loaded_structure.items():
        assert isinstance(subcategories, dict)
        assert len(subcategories) > 0

@then("each subcategory should have example sentences with DL labels")
def then_subcategories_have_sentences(when_parse_dl_structure):
    """Verify subcategories have sentences with labels."""
    steps = when_parse_dl_structure
    for category, subcategories in steps.loaded_structure.items():
        for subcategory, sentences in subcategories.items():
            assert isinstance(sentences, list)
            assert len(sentences) > 0
            for sentence in sentences:
                assert isinstance(sentence, str)
                assert len(sentence) > 0

@then("sentences should be mapped to their DL archetypes")
def then_sentences_mapped_to_archetypes(when_parse_dl_structure):
    """Verify sentences are mapped to DL archetypes."""
    steps = when_parse_dl_structure
    # This would be verified by the extraction process
    assert steps.loaded_structure is not None

@then("I should get a flat list of sentences")
def then_get_flat_list(when_extract_sentences_with_labels):
    """Verify flat list of sentences."""
    steps = when_extract_sentences_with_labels
    assert isinstance(steps.extracted_sentences, list)
    assert len(steps.extracted_sentences) > 0

@then("each sentence should have category metadata")
def then_sentences_have_category(when_extract_sentences_with_labels):
    """Verify sentences have category metadata."""
    steps = when_extract_sentences_with_labels
    for sentence in steps.extracted_sentences:
        assert 'dl_category' in sentence
        assert sentence['dl_category'] is not None
        assert sentence['dl_category'] != ""

@then("each sentence should have subcategory metadata")
def then_sentences_have_subcategory(when_extract_sentences_with_labels):
    """Verify sentences have subcategory metadata."""
    steps = when_extract_sentences_with_labels
    for sentence in steps.extracted_sentences:
        assert 'dl_subcategory' in sentence
        assert sentence['dl_subcategory'] is not None
        assert sentence['dl_subcategory'] != ""

@then("each sentence should have archetype metadata")
def then_sentences_have_archetype(when_extract_sentences_with_labels):
    """Verify sentences have archetype metadata."""
    steps = when_extract_sentences_with_labels
    for sentence in steps.extracted_sentences:
        assert 'dl_archetype' in sentence
        assert sentence['dl_archetype'] is not None
        assert sentence['dl_archetype'] != ""

@then("preserve the hierarchical DL relationship")
def then_preserve_hierarchical_relationship(when_extract_sentences_with_labels):
    """Verify hierarchical DL relationship is preserved."""
    steps = when_extract_sentences_with_labels
    for sentence in steps.extracted_sentences:
        # Verify archetype contains category and subcategory
        archetype = sentence['dl_archetype']
        category = sentence['dl_category']
        subcategory = sentence['dl_subcategory']
        assert category in archetype
        assert subcategory in archetype

@then("every sentence should have a non-empty category")
def then_every_sentence_has_category(when_validate_dl_completeness):
    """Verify every sentence has non-empty category."""
    steps = when_validate_dl_completeness
    assert steps.validation_results['categories_complete'] == True

@then("every sentence should have a non-empty subcategory")
def then_every_sentence_has_subcategory(when_validate_dl_completeness):
    """Verify every sentence has non-empty subcategory."""
    steps = when_validate_dl_completeness
    assert steps.validation_results['subcategories_complete'] == True

@then("every sentence should have a valid DL archetype")
def then_every_sentence_has_archetype(when_validate_dl_completeness):
    """Verify every sentence has valid DL archetype."""
    steps = when_validate_dl_completeness
    assert steps.validation_results['archetypes_complete'] == True

@then("no sentences should be missing DL metadata")
def then_no_missing_metadata(when_validate_dl_completeness):
    """Verify no sentences are missing DL metadata."""
    steps = when_validate_dl_completeness
    assert steps.validation_results['missing_metadata_count'] == 0

@then("I should get a warning about incomplete DL metadata")
def then_get_warning_about_incomplete_metadata(when_load_data_with_missing_labels):
    """Verify warning about incomplete metadata."""
    steps = when_load_data_with_missing_labels
    assert len(steps.warnings) > 0
    assert any("incomplete" in warning.lower() or "missing" in warning.lower() for warning in steps.warnings)

@then("sentences with complete labels should be processed")
def then_complete_labels_processed(when_load_data_with_missing_labels):
    """Verify sentences with complete labels are processed."""
    steps = when_load_data_with_missing_labels
    # Should have at least one valid sentence
    assert steps.extracted_sentences is not None
    valid_sentences = [s for s in steps.extracted_sentences if s is not None]
    assert len(valid_sentences) > 0

@then("sentences with missing labels should be flagged for review")
def then_missing_labels_flagged(when_load_data_with_missing_labels):
    """Verify sentences with missing labels are flagged."""
    steps = when_load_data_with_missing_labels
    assert len(steps.warnings) > 0
