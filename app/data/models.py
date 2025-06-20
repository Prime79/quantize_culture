"""Data models and schemas for the Digital Leadership Assessment pipeline."""

from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import json

@dataclass
class Sentence:
    """Represents a single sentence with metadata and DL labels."""
    id: str
    text: str
    archetype: str
    dimensions: List[str]
    contextualized_text: Optional[str] = None
    embedding: Optional[List[float]] = None
    cluster_id: Optional[int] = None
    
    # Enhanced DL metadata fields
    dl_category: Optional[str] = None
    dl_subcategory: Optional[str] = None
    dl_archetype: Optional[str] = None
    actual_phrase: Optional[str] = None
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Set actual_phrase from text if not provided
        if self.actual_phrase is None:
            self.actual_phrase = self.text
            
        # Auto-generate dl_archetype if category and subcategory are provided
        if self.dl_archetype is None and self.dl_category and self.dl_subcategory:
            self.dl_archetype = f"{self.dl_category} - {self.dl_subcategory}"
    
    def has_complete_dl_metadata(self) -> bool:
        """Check if sentence has complete DL metadata."""
        return all([
            self.dl_category is not None and self.dl_category != "",
            self.dl_subcategory is not None and self.dl_subcategory != "",
            self.dl_archetype is not None and self.dl_archetype != ""
        ])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'id': self.id,
            'text': self.text,
            'archetype': self.archetype,
            'dimensions': self.dimensions,
            'contextualized_text': self.contextualized_text,
            'embedding': self.embedding,
            'cluster_id': self.cluster_id,
            'dl_category': self.dl_category,
            'dl_subcategory': self.dl_subcategory,
            'dl_archetype': self.dl_archetype,
            'actual_phrase': self.actual_phrase
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Sentence':
        """Create from dictionary representation."""
        return cls(
            id=data['id'],
            text=data['text'],
            archetype=data['archetype'],
            dimensions=data['dimensions'],
            contextualized_text=data.get('contextualized_text'),
            embedding=data.get('embedding'),
            cluster_id=data.get('cluster_id')
        )

@dataclass
class Archetype:
    """Represents a leadership archetype."""
    name: str
    description: str
    dimensions: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'name': self.name,
            'description': self.description,
            'dimensions': self.dimensions
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Archetype':
        """Create from dictionary representation."""
        return cls(
            name=data['name'],
            description=data['description'],
            dimensions=data['dimensions']
        )

@dataclass
class DLReference:
    """Represents a complete Digital Leadership reference dataset."""
    archetypes: List[Archetype]
    sentences: List[Sentence]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'archetypes': [archetype.to_dict() for archetype in self.archetypes],
            'sentences': [sentence.to_dict() for sentence in self.sentences]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DLReference':
        """Create from dictionary representation."""
        archetypes = [Archetype.from_dict(a) for a in data['archetypes']]
        sentences = [Sentence.from_dict(s) for s in data['sentences']]
        return cls(archetypes=archetypes, sentences=sentences)
    
    @classmethod
    def from_json_file(cls, file_path: str) -> 'DLReference':
        """Load from JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def to_json_file(self, file_path: str) -> None:
        """Save to JSON file."""
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

@dataclass
class ClusterResult:
    """Represents clustering analysis results."""
    cluster_labels: List[int]
    n_clusters: int
    noise_points: int
    cluster_summary: Dict[int, Dict[str, Any]]
    quality_metrics: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'cluster_labels': self.cluster_labels,
            'n_clusters': self.n_clusters,
            'noise_points': self.noise_points,
            'cluster_summary': self.cluster_summary,
            'quality_metrics': self.quality_metrics
        }

@dataclass
class AssessmentResult:
    """Represents qualitative assessment results."""
    semantic_coherence: float
    cultural_alignment: float
    business_interpretability: float
    actionable_insights: Dict[str, Any]
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'semantic_coherence': self.semantic_coherence,
            'cultural_alignment': self.cultural_alignment,
            'business_interpretability': self.business_interpretability,
            'actionable_insights': self.actionable_insights,
            'recommendations': self.recommendations
        }

@dataclass
class DLSentence:
    """
    Specialized sentence model for Digital Leadership with required DL metadata.
    """
    text: str
    dl_category: str
    dl_subcategory: str
    dl_archetype: Optional[str] = None
    id: Optional[str] = None
    contextualized_text: Optional[str] = None
    embedding: Optional[List[float]] = None
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.dl_archetype is None:
            self.dl_archetype = f"{self.dl_category} - {self.dl_subcategory}"
        
        if self.id is None:
            import uuid
            self.id = str(uuid.uuid4())
    
    def to_sentence(self) -> Sentence:
        """Convert to standard Sentence object."""
        return Sentence(
            id=self.id,
            text=self.text,
            archetype=self.dl_archetype,
            dimensions=[],
            dl_category=self.dl_category,
            dl_subcategory=self.dl_subcategory,
            dl_archetype=self.dl_archetype,
            contextualized_text=self.contextualized_text,
            embedding=self.embedding,
            actual_phrase=self.text
        )

@dataclass
class DLDataStructure:
    """
    Represents the hierarchical structure of DL data.
    """
    categories: Dict[str, Dict[str, List[str]]] = None
    total_sentences: int = 0
    total_categories: int = 0
    total_subcategories: int = 0
    
    def __post_init__(self):
        """Calculate totals after initialization."""
        if self.categories is None:
            self.categories = {}
        
        self.total_categories = len(self.categories)
        self.total_subcategories = sum(len(subcats) for subcats in self.categories.values())
        self.total_sentences = sum(
            len(sentences) 
            for subcats in self.categories.values() 
            for sentences in subcats.values()
        )
    
    def get_all_sentences_with_metadata(self) -> List[DLSentence]:
        """Extract all sentences with their DL metadata."""
        sentences = []
        for category, subcategories in self.categories.items():
            for subcategory, sentence_texts in subcategories.items():
                for text in sentence_texts:
                    sentence = DLSentence(
                        text=text,
                        dl_category=category,
                        dl_subcategory=subcategory
                    )
                    sentences.append(sentence)
        return sentences

@dataclass
class DLValidationResult:
    """
    Results of DL metadata validation.
    """
    categories_complete: bool = True
    subcategories_complete: bool = True
    archetypes_complete: bool = True
    missing_metadata_count: int = 0
    validation_errors: List[str] = None
    total_sentences: int = 0
    valid_sentences: int = 0
    
    def __post_init__(self):
        """Initialize validation errors list."""
        if self.validation_errors is None:
            self.validation_errors = []
    
    @property
    def is_valid(self) -> bool:
        """Check if validation passed."""
        return (self.categories_complete and 
                self.subcategories_complete and 
                self.archetypes_complete and 
                self.missing_metadata_count == 0)
    
    @property
    def completion_rate(self) -> float:
        """Calculate completion rate."""
        if self.total_sentences == 0:
            return 0.0
        return self.valid_sentences / self.total_sentences
