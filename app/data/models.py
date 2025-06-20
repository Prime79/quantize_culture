"""Data models and schemas for the Digital Leadership Assessment pipeline."""

from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import json

@dataclass
class Sentence:
    """Represents a single sentence with metadata."""
    id: str
    text: str
    archetype: str
    dimensions: List[str]
    contextualized_text: Optional[str] = None
    embedding: Optional[List[float]] = None
    cluster_id: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'id': self.id,
            'text': self.text,
            'archetype': self.archetype,
            'dimensions': self.dimensions,
            'contextualized_text': self.contextualized_text,
            'embedding': self.embedding,
            'cluster_id': self.cluster_id
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
