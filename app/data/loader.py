"""Data loading functionality for the Digital Leadership Assessment pipeline."""

import json
import os
import uuid
from typing import List, Optional, Dict, Any
from ..data.models import DLReference, Sentence, Archetype, DLSentence, DLDataStructure, DLValidationResult

class DataLoader:
    """Handles loading and parsing of DL reference data."""
    
    def load_from_json(self, file_path: str) -> DLReference:
        """
        Load DL reference data from JSON file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            DLReference object containing archetypes and sentences
        """
        try:
            return DLReference.from_json_file(file_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {file_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in {file_path}: {str(e)}")
        except Exception as e:
            raise Exception(f"Failed to load data from {file_path}: {str(e)}")
    
    def validate_data(self, dl_reference: DLReference) -> bool:
        """
        Validate that the loaded data is well-formed.
        
        Args:
            dl_reference: DL reference data to validate
            
        Returns:
            True if valid, raises exception if invalid
        """
        # Validate archetypes
        if not dl_reference.archetypes:
            raise ValueError("No archetypes found in data")
        
        archetype_names = set()
        for archetype in dl_reference.archetypes:
            if not archetype.name:
                raise ValueError("Archetype missing name")
            if archetype.name in archetype_names:
                raise ValueError(f"Duplicate archetype name: {archetype.name}")
            archetype_names.add(archetype.name)
            
            if not archetype.dimensions:
                raise ValueError(f"Archetype '{archetype.name}' has no dimensions")
        
        # Validate sentences
        if not dl_reference.sentences:
            raise ValueError("No sentences found in data")
        
        sentence_ids = set()
        for sentence in dl_reference.sentences:
            if not sentence.id:
                raise ValueError("Sentence missing ID")
            if sentence.id in sentence_ids:
                raise ValueError(f"Duplicate sentence ID: {sentence.id}")
            sentence_ids.add(sentence.id)
            
            if not sentence.text:
                raise ValueError(f"Sentence '{sentence.id}' has no text")
            
            if sentence.archetype not in archetype_names:
                raise ValueError(f"Sentence '{sentence.id}' references unknown archetype: {sentence.archetype}")
            
            if not sentence.dimensions:
                raise ValueError(f"Sentence '{sentence.id}' has no dimensions")
        
        return True
    
    def get_sentences_by_archetype(self, dl_reference: DLReference, 
                                  archetype_name: str) -> List[Sentence]:
        """
        Get all sentences for a specific archetype.
        
        Args:
            dl_reference: DL reference data
            archetype_name: Name of the archetype
            
        Returns:
            List of sentences for the archetype
        """
        return [s for s in dl_reference.sentences if s.archetype == archetype_name]
    
    def get_archetype_by_name(self, dl_reference: DLReference, 
                             archetype_name: str) -> Optional[Archetype]:
        """
        Get archetype by name.
        
        Args:
            dl_reference: DL reference data
            archetype_name: Name of the archetype
            
        Returns:
            Archetype object or None if not found
        """
        for archetype in dl_reference.archetypes:
            if archetype.name == archetype_name:
                return archetype
        return None

class DLDataLoader:
    """
    Enhanced data loader for Digital Leadership sentences with DL archetype labels.
    """
    
    def __init__(self):
        """Initialize the DL data loader."""
        self.warnings = []
    
    def load_dl_structure(self, data: Dict[str, Any]) -> DLDataStructure:
        """
        Load DL data structure from dictionary.
        
        Args:
            data: Dictionary with DL categories and subcategories
            
        Returns:
            DLDataStructure object
        """
        return DLDataStructure(categories=data)
    
    def parse_dl_archetypes(self, data: Dict[str, Any]) -> DLDataStructure:
        """
        Parse DL archetype structure from source data.
        
        Args:
            data: Source data dictionary
            
        Returns:
            Parsed DL structure
        """
        return DLDataStructure(categories=data)
    
    def extract_sentences_with_dl_metadata(self, data: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Extract sentences with their DL metadata.
        
        Args:
            data: Source data with DL categories and subcategories
            
        Returns:
            List of dictionaries with sentence text and DL metadata
        """
        sentences = []
        self.warnings = []
        
        for category, subcategories in data.items():
            if not isinstance(subcategories, dict):
                self.warnings.append(f"Category '{category}' has invalid subcategory structure")
                continue
                
            for subcategory, sentence_list in subcategories.items():
                if not isinstance(sentence_list, list):
                    self.warnings.append(f"Subcategory '{subcategory}' has invalid sentence list")
                    continue
                
                for sentence_text in sentence_list:
                    if sentence_text is None or sentence_text == "":
                        self.warnings.append(f"Found empty sentence in {category} - {subcategory}")
                        continue
                    
                    sentence_data = {
                        'text': sentence_text,
                        'dl_category': category,
                        'dl_subcategory': subcategory,
                        'dl_archetype': f"{category} - {subcategory}"
                    }
                    sentences.append(sentence_data)
        
        return sentences
    
    def validate_dl_completeness(self, sentences: List[Dict[str, str]]) -> DLValidationResult:
        """
        Validate that all sentences have complete DL metadata.
        
        Args:
            sentences: List of sentence dictionaries with DL metadata
            
        Returns:
            Validation result
        """
        result = DLValidationResult()
        result.total_sentences = len(sentences)
        
        missing_categories = 0
        missing_subcategories = 0
        missing_archetypes = 0
        
        for sentence in sentences:
            has_category = sentence.get('dl_category') is not None and sentence.get('dl_category') != ""
            has_subcategory = sentence.get('dl_subcategory') is not None and sentence.get('dl_subcategory') != ""
            has_archetype = sentence.get('dl_archetype') is not None and sentence.get('dl_archetype') != ""
            
            if not has_category:
                missing_categories += 1
            if not has_subcategory:
                missing_subcategories += 1
            if not has_archetype:
                missing_archetypes += 1
            
            if has_category and has_subcategory and has_archetype:
                result.valid_sentences += 1
        
        result.categories_complete = missing_categories == 0
        result.subcategories_complete = missing_subcategories == 0
        result.archetypes_complete = missing_archetypes == 0
        result.missing_metadata_count = missing_categories + missing_subcategories + missing_archetypes
        
        if not result.categories_complete:
            result.validation_errors.append(f"Missing categories in {missing_categories} sentences")
        if not result.subcategories_complete:
            result.validation_errors.append(f"Missing subcategories in {missing_subcategories} sentences")
        if not result.archetypes_complete:
            result.validation_errors.append(f"Missing archetypes in {missing_archetypes} sentences")
        
        return result
    
    def get_warnings(self) -> List[str]:
        """Get warnings from the last operation."""
        return self.warnings
    
    def load_from_json_file(self, file_path: str) -> DLDataStructure:
        """
        Load DL data structure from JSON file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            DL data structure
            
        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If file contains invalid JSON
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"DL data file not found: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return self.parse_dl_archetypes(data)
            
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Invalid JSON in DL data file: {str(e)}", e.doc, e.pos)
    
    def convert_to_sentences(self, dl_structure: DLDataStructure) -> List[Sentence]:
        """
        Convert DL structure to list of Sentence objects.
        
        Args:
            dl_structure: DL data structure
            
        Returns:
            List of Sentence objects with DL metadata
        """
        sentences = []
        
        for category, subcategories in dl_structure.categories.items():
            for subcategory, sentence_texts in subcategories.items():
                for text in sentence_texts:
                    sentence = Sentence(
                        id=str(uuid.uuid4()),
                        text=text,
                        archetype=f"{category} - {subcategory}",
                        dimensions=[],
                        dl_category=category,
                        dl_subcategory=subcategory,
                        dl_archetype=f"{category} - {subcategory}",
                        actual_phrase=text
                    )
                    sentences.append(sentence)
        
        return sentences
    
    def create_contextualized_sentences(self, sentences: List[Sentence], 
                                      context_prefix: str = "Domain Logic example phrase: ") -> List[Sentence]:
        """
        Create contextualized versions of sentences.
        
        Args:
            sentences: List of sentences
            context_prefix: Prefix to add for contextualization
            
        Returns:
            List of sentences with contextualized text
        """
        for sentence in sentences:
            sentence.contextualized_text = f"{context_prefix}{sentence.text}"
        
        return sentences
