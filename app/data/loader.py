"""Data loading functionality for the Digital Leadership Assessment pipeline."""

import json
from typing import List, Optional
from ..data.models import DLReference, Sentence, Archetype

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
