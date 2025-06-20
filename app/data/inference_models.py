"""Enhanced data models for sophisticated DL inference with multi-factor confidence assessment."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from enum import Enum
from datetime import datetime

class ConfidenceLevel(Enum):
    STRONG = "STRONG"
    WEAK = "WEAK" 
    AMBIGUOUS = "AMBIGUOUS"
    TRAINING_MATCH = "TRAINING_MATCH"
    NO_MATCH = "NO_MATCH"

class ClassificationStatus(Enum):
    STRONG_MATCH = "STRONG_MATCH"
    WEAK_MATCH = "WEAK_MATCH"
    MULTIPLE_GOOD_MATCHES = "MULTIPLE_GOOD_MATCHES"
    AMBIGUOUS_MATCH = "AMBIGUOUS_MATCH"
    TRAINING_LEAKAGE = "TRAINING_LEAKAGE"
    NO_MATCH = "NO_MATCH"

@dataclass
class ClusterStatistics:
    """Statistics about clusters for adaptive confidence assessment"""
    avg_intra_cluster_similarity: float
    avg_inter_cluster_distance: float
    cluster_tightness_score: float
    cluster_sizes: List[int]
    cluster_quality_scores: List[float]
    similarity_std_dev: float
    
    def to_dict(self) -> Dict:
        return {
            'avg_intra_cluster_similarity': self.avg_intra_cluster_similarity,
            'avg_inter_cluster_distance': self.avg_inter_cluster_distance,
            'cluster_tightness_score': self.cluster_tightness_score,
            'cluster_sizes': self.cluster_sizes,
            'cluster_quality_scores': self.cluster_quality_scores,
            'similarity_std_dev': self.similarity_std_dev
        }
    
@dataclass
class AdaptiveThresholds:
    """Adaptive confidence thresholds based on cluster characteristics"""
    strong_threshold: float
    weak_threshold: float
    ambiguity_gap_threshold: float  # Max gap for ambiguous classification
    training_leakage_threshold: float
    
    def to_dict(self) -> Dict:
        return {
            'strong_threshold': self.strong_threshold,
            'weak_threshold': self.weak_threshold,
            'ambiguity_gap_threshold': self.ambiguity_gap_threshold,
            'training_leakage_threshold': self.training_leakage_threshold
        }
    
@dataclass
class SemanticAnalysis:
    """Results of semantic similarity analysis beyond vector similarity"""
    keyword_overlap_score: float
    domain_terminology_present: bool
    semantic_coherence_score: float
    terminology_boost: float
    matched_keywords: List[str]
    missing_keywords: List[str]
    
    def to_dict(self) -> Dict:
        return {
            'keyword_overlap_score': self.keyword_overlap_score,
            'domain_terminology_present': self.domain_terminology_present,
            'semantic_coherence_score': self.semantic_coherence_score,
            'terminology_boost': self.terminology_boost,
            'matched_keywords': self.matched_keywords,
            'missing_keywords': self.missing_keywords
        }

@dataclass
class MultiFactorConfidence:
    """Comprehensive confidence assessment considering multiple factors"""
    base_similarity: float
    percentile_rank: float
    gap_to_second: float
    cluster_size_factor: float
    cluster_quality_factor: float
    semantic_boost: float
    model_compatibility_penalty: float
    final_confidence: float
    
    def to_dict(self) -> Dict:
        return {
            'base_similarity': self.base_similarity,
            'percentile_rank': self.percentile_rank,
            'gap_to_second': self.gap_to_second,
            'cluster_size_factor': self.cluster_size_factor,
            'cluster_quality_factor': self.cluster_quality_factor,
            'semantic_boost': self.semantic_boost,
            'model_compatibility_penalty': self.model_compatibility_penalty,
            'final_confidence': self.final_confidence
        }

@dataclass
class InferenceMatch:
    """Single inference match result"""
    cluster_id: int
    archetype: str
    similarity_score: float
    adjusted_confidence: float
    confidence_level: ConfidenceLevel
    reliability_score: float
    semantic_analysis: SemanticAnalysis
    
    def to_dict(self) -> Dict:
        return {
            'cluster_id': self.cluster_id,
            'archetype': self.archetype,
            'similarity_score': self.similarity_score,
            'adjusted_confidence': self.adjusted_confidence,
            'confidence_level': self.confidence_level.value,
            'reliability_score': self.reliability_score,
            'semantic_analysis': self.semantic_analysis.to_dict()
        }

@dataclass
class EnhancedInferenceResult:
    """Comprehensive inference result with all analysis"""
    input_sentence: str
    contextualized_sentence: str
    primary_match: InferenceMatch
    alternative_matches: List[InferenceMatch]
    classification_status: ClassificationStatus
    multi_factor_confidence: MultiFactorConfidence
    warnings: List[str]
    recommendations: List[str]
    model_compatibility_score: float
    training_leakage_detected: bool
    processing_timestamp: str
    
    def to_dict(self) -> Dict:
        return {
            'input_sentence': self.input_sentence,
            'contextualized_sentence': self.contextualized_sentence,
            'primary_match': self.primary_match.to_dict(),
            'alternative_matches': [match.to_dict() for match in self.alternative_matches],
            'classification_status': self.classification_status.value,
            'multi_factor_confidence': self.multi_factor_confidence.to_dict(),
            'warnings': self.warnings,
            'recommendations': self.recommendations,
            'model_compatibility_score': self.model_compatibility_score,
            'training_leakage_detected': self.training_leakage_detected,
            'processing_timestamp': self.processing_timestamp
        }

@dataclass
class EnhancedReferenceMetadata:
    """Enhanced reference database metadata with statistics and thresholds"""
    context_phrase: str
    model_name: str
    cluster_centroids: Dict[int, Dict]
    cluster_statistics: ClusterStatistics  
    adaptive_thresholds: AdaptiveThresholds
    training_sentences: List[str]  # For leakage detection
    domain_keywords: Dict[int, List[str]]  # Cluster-specific keywords
    creation_timestamp: str
    version: str
    
    def to_dict(self) -> Dict:
        return {
            'context_phrase': self.context_phrase,
            'model_name': self.model_name,
            'cluster_centroids': self.cluster_centroids,
            'cluster_statistics': self.cluster_statistics.to_dict(),
            'adaptive_thresholds': self.adaptive_thresholds.to_dict(),
            'training_sentences': self.training_sentences,
            'domain_keywords': self.domain_keywords,
            'creation_timestamp': self.creation_timestamp,
            'version': self.version
        }
