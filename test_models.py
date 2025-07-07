"""Simplified test of inference models."""

from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum

class ConfidenceLevel(Enum):
    STRONG = "STRONG"
    WEAK = "WEAK" 
    AMBIGUOUS = "AMBIGUOUS"
    TRAINING_MATCH = "TRAINING_MATCH"
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

@dataclass
class AdaptiveThresholds:
    """Adaptive confidence thresholds based on cluster characteristics"""
    strong_threshold: float
    weak_threshold: float
    ambiguity_gap_threshold: float
    training_leakage_threshold: float

@dataclass
class EnhancedReferenceMetadata:
    """Enhanced reference database metadata with statistics and thresholds"""
    context_phrase: str
    model_name: str
    cluster_centroids: Dict[int, Dict]
    cluster_statistics: ClusterStatistics  
    adaptive_thresholds: AdaptiveThresholds
    training_sentences: List[str]
    domain_keywords: Dict[int, List[str]]
    creation_timestamp: str
    version: str
