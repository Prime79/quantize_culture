"""Enhanced Digital Leadership Inference Engine with sophisticated confidence assessment."""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
import re
from sklearn.metrics.pairwise import cosine_similarity
from ..data.inference_models import *
from ..services.openai_client import OpenAIService
from ..services.qdrant_client import QdrantService
from ..core.contextualization import ContextualizationService
from ..utils.config import config

class EnhancedInferenceEngine:
    """
    Advanced inference engine with multi-factor confidence assessment,
    semantic analysis, and adaptive thresholds.
    """
    
    def __init__(self, openai_service: Optional[OpenAIService] = None,
                 qdrant_service: Optional[QdrantService] = None):
        """Initialize enhanced inference engine."""
        self.openai_service = openai_service or OpenAIService()
        self.qdrant_service = qdrant_service or QdrantService()
        self.contextualization_service = ContextualizationService()
        
        # Configuration for inference
        self.min_cluster_size_for_reliability = 10
        self.confidence_decay_factor = 0.1
        self.max_alternative_matches = 5
        self.semantic_boost_factor = 0.05
    
    def load_reference_database_metadata(self, collection_name: str) -> EnhancedReferenceMetadata:
        """
        Load enhanced metadata for a reference database.
        
        Args:
            collection_name: Name of the reference collection
            
        Returns:
            Enhanced metadata with statistics and adaptive thresholds
        """
        try:
            # Extract data from Qdrant
            sentences = self.qdrant_service.extract_data(collection_name)
            
            if not sentences:
                raise ValueError(f"No data found in collection '{collection_name}'")
            
            # Calculate cluster statistics
            cluster_stats = self._calculate_cluster_statistics(sentences)
            
            # Compute adaptive thresholds
            adaptive_thresholds = self._compute_adaptive_thresholds(cluster_stats)
            
            # Extract cluster centroids and metadata
            cluster_centroids = self._extract_cluster_centroids(sentences)
            
            # Build domain keywords for each cluster
            domain_keywords = self._extract_domain_keywords(sentences)
            
            # Extract training sentences for leakage detection
            training_sentences = [s.text for s in sentences]
            
            # Get context phrase from first sentence
            context_phrase = self._extract_context_phrase(sentences)
            
            metadata = EnhancedReferenceMetadata(
                context_phrase=context_phrase,
                model_name=config.embedding_model,
                cluster_centroids=cluster_centroids,
                cluster_statistics=cluster_stats,
                adaptive_thresholds=adaptive_thresholds,
                training_sentences=training_sentences,
                domain_keywords=domain_keywords,
                creation_timestamp=datetime.now().isoformat(),
                version="1.0"
            )
            
            return metadata
            
        except Exception as e:
            raise Exception(f"Failed to load reference database metadata: {str(e)}")
    
    def infer_dl_archetype(self, sentence: str, collection_name: str) -> EnhancedInferenceResult:
        """
        Perform sophisticated DL archetype inference with multi-factor confidence assessment.
        
        Args:
            sentence: Input sentence to classify
            collection_name: Reference database collection
            
        Returns:
            Comprehensive inference result
        """
        try:
            # Validate input
            self._validate_input_sentence(sentence)
            
            # Load reference metadata
            metadata = self.load_reference_database_metadata(collection_name)
            
            # Check model compatibility
            model_compatibility_score = self._check_model_compatibility(metadata.model_name)
            
            # Contextualize sentence
            self.contextualization_service.set_context_prefix(metadata.context_phrase)
            contextualized_sentence = self.contextualization_service.contextualize_text(sentence)
            
            # Generate embedding
            embedding = self.openai_service.get_embedding(contextualized_sentence)
            
            # Check for training data leakage
            leakage_detected, leakage_similarity = self._detect_training_leakage(
                sentence, metadata.training_sentences
            )
            
            # Calculate similarities to all clusters
            similarities = self._calculate_all_similarities(embedding, metadata.cluster_centroids)
            
            # Perform multi-factor confidence assessment
            multi_factor_confidence = self._calculate_multi_factor_confidence(
                similarities, metadata.cluster_statistics, metadata.adaptive_thresholds
            )
            
            # Semantic analysis
            semantic_analyses = self._perform_semantic_analysis(
                sentence, metadata.domain_keywords, similarities
            )
            
            # Create inference matches
            primary_match, alternative_matches = self._create_inference_matches(
                similarities, semantic_analyses, metadata, multi_factor_confidence
            )
            
            # Determine classification status
            classification_status = self._determine_classification_status(
                primary_match, alternative_matches, leakage_detected, multi_factor_confidence
            )
            
            # Generate warnings and recommendations
            warnings, recommendations = self._generate_warnings_and_recommendations(
                primary_match, alternative_matches, model_compatibility_score,
                leakage_detected, metadata.adaptive_thresholds
            )
            
            result = EnhancedInferenceResult(
                input_sentence=sentence,
                contextualized_sentence=contextualized_sentence,
                primary_match=primary_match,
                alternative_matches=alternative_matches,
                classification_status=classification_status,
                multi_factor_confidence=multi_factor_confidence,
                warnings=warnings,
                recommendations=recommendations,
                model_compatibility_score=model_compatibility_score,
                training_leakage_detected=leakage_detected,
                processing_timestamp=datetime.now().isoformat()
            )
            
            return result
            
        except Exception as e:
            raise Exception(f"Inference failed: {str(e)}")
    
    def get_nearest_neighbors(self, sentence: str, collection_name: str, 
                            n_neighbors: int = 5, 
                            similarity_threshold: Optional[float] = None,
                            include_dominant_logic: bool = True) -> Dict[str, Any]:
        """
        Find the n nearest neighbor sentences to the input sentence with DL metadata analysis.
        
        Args:
            sentence: Input sentence to find neighbors for
            collection_name: Qdrant collection to search in
            n_neighbors: Number of nearest neighbors to return (default: 5)
            similarity_threshold: Minimum similarity threshold (optional)
            include_dominant_logic: Whether to include dominant logic analysis
            
        Returns:
            Dictionary containing neighbors, similarity scores, DL metadata, and analysis
        """
        try:
            # Validate input
            self._validate_input_sentence(sentence)
            self._validate_neighbors_parameters(n_neighbors, similarity_threshold)
            
            # Check if collection exists
            if not self.qdrant_service.collection_exists(collection_name):
                raise ValueError(f"Collection '{collection_name}' not found")
            
            # Contextualize and embed the input sentence
            context_phrase = "Domain Logic example phrase:"  # Default context
            contextualized_sentence = f"{context_phrase} {sentence}"
            embedding = self.openai_service.get_embedding(contextualized_sentence)
            
            # Search for similar points in the vector database
            search_results = self.qdrant_service.search_similar(
                query_vector=embedding,
                collection_name=collection_name,
                limit=n_neighbors * 2  # Get more to apply threshold filtering
            )
            
            # Extract and process neighbor data
            neighbors = self._process_search_results(
                search_results, n_neighbors, similarity_threshold
            )
            
            # Extract DL metadata from neighbors
            neighbors_with_metadata = self._extract_neighbor_dl_metadata(neighbors)
            
            # Analyze dominant logic if requested
            dominant_logic_analysis = None
            distribution_stats = None
            
            if include_dominant_logic and neighbors_with_metadata:
                dominant_logic_analysis = self._analyze_neighbor_dominant_logic(neighbors_with_metadata)
                distribution_stats = self._calculate_neighbor_distribution_stats(neighbors_with_metadata)
            
            # Format result
            result = {
                'query_sentence': sentence,
                'contextualized_sentence': contextualized_sentence,
                'collection_name': collection_name,
                'neighbors': neighbors_with_metadata,
                'total_neighbors': len(neighbors_with_metadata),
                'similarity_threshold': similarity_threshold,
                'timestamp': datetime.now().isoformat()
            }
            
            if dominant_logic_analysis:
                result['dominant_logic'] = dominant_logic_analysis
                
            if distribution_stats:
                result['distribution_stats'] = distribution_stats
            
            return result
            
        except Exception as e:
            raise Exception(f"Nearest neighbors search failed: {str(e)}")
    
    def _calculate_cluster_statistics(self, sentences: List) -> ClusterStatistics:
        """Calculate comprehensive cluster statistics for adaptive thresholds."""
        try:
            # Group sentences by cluster
            clusters = {}
            for sentence in sentences:
                if sentence.cluster_id is not None:
                    if sentence.cluster_id not in clusters:
                        clusters[sentence.cluster_id] = []
                    clusters[sentence.cluster_id].append(sentence.embedding)
            
            if not clusters:
                raise ValueError("No clustered sentences found")
            
            # Calculate intra-cluster similarities
            intra_cluster_sims = []
            cluster_sizes = []
            cluster_quality_scores = []
            
            for cluster_id, embeddings in clusters.items():
                if len(embeddings) > 1:
                    # Calculate pairwise similarities within cluster
                    sims = cosine_similarity(embeddings)
                    # Get upper triangle (excluding diagonal)
                    upper_triangle = sims[np.triu_indices_from(sims, k=1)]
                    intra_cluster_sims.extend(upper_triangle)
                    
                    # Cluster quality = average intra-cluster similarity
                    cluster_quality_scores.append(np.mean(upper_triangle))
                else:
                    cluster_quality_scores.append(0.5)  # Default for single-sentence clusters
                
                cluster_sizes.append(len(embeddings))
            
            # Calculate inter-cluster distances
            inter_cluster_distances = []
            cluster_ids = list(clusters.keys())
            for i in range(len(cluster_ids)):
                for j in range(i + 1, len(cluster_ids)):
                    cluster1_embeddings = clusters[cluster_ids[i]]
                    cluster2_embeddings = clusters[cluster_ids[j]]
                    
                    # Calculate centroid distance
                    centroid1 = np.mean(cluster1_embeddings, axis=0)
                    centroid2 = np.mean(cluster2_embeddings, axis=0)
                    distance = 1 - cosine_similarity([centroid1], [centroid2])[0][0]
                    inter_cluster_distances.append(distance)
            
            avg_intra_similarity = np.mean(intra_cluster_sims) if intra_cluster_sims else 0.5
            avg_inter_distance = np.mean(inter_cluster_distances) if inter_cluster_distances else 0.5
            similarity_std_dev = np.std(intra_cluster_sims) if intra_cluster_sims else 0.15
            
            # Cluster tightness = ratio of intra-cluster similarity to inter-cluster distance
            cluster_tightness = avg_intra_similarity / max(avg_inter_distance, 0.1)
            
            return ClusterStatistics(
                avg_intra_cluster_similarity=avg_intra_similarity,
                avg_inter_cluster_distance=avg_inter_distance,
                cluster_tightness_score=min(cluster_tightness, 1.0),
                cluster_sizes=cluster_sizes,
                cluster_quality_scores=cluster_quality_scores,
                similarity_std_dev=similarity_std_dev
            )
            
        except Exception as e:
            # Return default statistics if calculation fails
            return ClusterStatistics(
                avg_intra_cluster_similarity=0.75,
                avg_inter_cluster_distance=0.5,
                cluster_tightness_score=0.6,
                cluster_sizes=[],
                cluster_quality_scores=[],
                similarity_std_dev=0.15
            )
    
    def _compute_adaptive_thresholds(self, cluster_stats: ClusterStatistics) -> AdaptiveThresholds:
        """Compute adaptive confidence thresholds based on cluster characteristics."""
        mean_sim = cluster_stats.avg_intra_cluster_similarity
        std_dev = cluster_stats.similarity_std_dev
        
        # Adaptive thresholds based on cluster statistics
        strong_threshold = min(mean_sim + 0.5 * std_dev, 0.95)
        weak_threshold = max(mean_sim - 0.5 * std_dev, 0.5)
        
        # Adjust for cluster tightness
        if cluster_stats.cluster_tightness_score > 0.8:
            # Very tight clusters - increase thresholds
            strong_threshold = min(strong_threshold + 0.1, 0.95)
            weak_threshold = min(weak_threshold + 0.05, strong_threshold - 0.1)
        
        return AdaptiveThresholds(
            strong_threshold=strong_threshold,
            weak_threshold=weak_threshold,
            ambiguity_gap_threshold=0.05,
            training_leakage_threshold=0.99
        )
    
    def _calculate_all_similarities(self, embedding: List[float], 
                                  cluster_centroids: Dict[int, Dict]) -> List[Tuple[int, str, float]]:
        """Calculate cosine similarities to all cluster centroids."""
        similarities = []
        
        for cluster_id, cluster_data in cluster_centroids.items():
            centroid = cluster_data["centroid"]
            archetype = cluster_data["archetype"]
            
            # Calculate cosine similarity
            similarity = cosine_similarity([embedding], [centroid])[0][0]
            similarities.append((cluster_id, archetype, similarity))
        
        # Sort by similarity (descending)
        return sorted(similarities, key=lambda x: x[2], reverse=True)
    
    def _validate_input_sentence(self, sentence: str) -> None:
        """Validate input sentence."""
        if not sentence or not sentence.strip():
            raise ValueError("Sentence cannot be empty")
        
        if len(sentence) > 2000:
            raise ValueError("Sentence too long (max 2000 characters)")
    
    def _check_model_compatibility(self, reference_model: str) -> float:
        """Check compatibility between reference and inference models."""
        current_model = config.embedding_model
        
        if reference_model == current_model:
            return 1.0
        
        # Define compatibility matrix
        compatibility_matrix = {
            ("text-embedding-ada-002", "text-embedding-3-small"): 0.8,
            ("text-embedding-3-small", "text-embedding-ada-002"): 0.8,
            ("text-embedding-3-large", "text-embedding-3-small"): 0.9,
            ("text-embedding-3-small", "text-embedding-3-large"): 0.9,
        }
        
        return compatibility_matrix.get((reference_model, current_model), 0.6)
    
    def _detect_training_leakage(self, sentence: str, training_sentences: List[str]) -> Tuple[bool, float]:
        """Detect potential training data leakage by checking similarity to training sentences."""
        max_similarity = 0.0
        
        for training_sentence in training_sentences:
            # Simple string similarity check first
            if sentence.lower().strip() == training_sentence.lower().strip():
                return True, 1.0
            
            # Calculate token-based similarity for more nuanced detection
            sentence_tokens = set(sentence.lower().split())
            training_tokens = set(training_sentence.lower().split())
            
            if sentence_tokens and training_tokens:
                jaccard_similarity = len(sentence_tokens & training_tokens) / len(sentence_tokens | training_tokens)
                max_similarity = max(max_similarity, jaccard_similarity)
        
        # Consider high token similarity as potential leakage
        leakage_detected = max_similarity > 0.95
        return leakage_detected, max_similarity
    
    def _extract_cluster_centroids(self, sentences: List) -> Dict[int, Dict]:
        """Extract cluster centroids and archetype mappings."""
        clusters = {}
        
        for sentence in sentences:
            if sentence.cluster_id is not None:
                if sentence.cluster_id not in clusters:
                    clusters[sentence.cluster_id] = {
                        "embeddings": [],
                        "archetype": sentence.archetype
                    }
                clusters[sentence.cluster_id]["embeddings"].append(sentence.embedding)
        
        # Calculate centroids
        cluster_centroids = {}
        for cluster_id, data in clusters.items():
            centroid = np.mean(data["embeddings"], axis=0).tolist()
            cluster_centroids[cluster_id] = {
                "centroid": centroid,
                "archetype": data["archetype"]
            }
        
        return cluster_centroids
    
    def _extract_domain_keywords(self, sentences: List) -> Dict[int, List[str]]:
        """Extract domain-specific keywords for each cluster."""
        cluster_keywords = {}
        
        for sentence in sentences:
            if sentence.cluster_id is not None:
                if sentence.cluster_id not in cluster_keywords:
                    cluster_keywords[sentence.cluster_id] = []
                
                # Extract keywords from sentence text
                words = re.findall(r'\b[a-zA-Z]{4,}\b', sentence.text.lower())
                
                # Filter for domain-relevant terms
                domain_terms = [word for word in words if self._is_domain_relevant(word)]
                cluster_keywords[sentence.cluster_id].extend(domain_terms)
        
        # Remove duplicates and get top keywords per cluster
        for cluster_id in cluster_keywords:
            cluster_keywords[cluster_id] = list(set(cluster_keywords[cluster_id]))[:20]
        
        return cluster_keywords
    
    def _is_domain_relevant(self, word: str) -> bool:
        """Check if a word is relevant to digital leadership domain."""
        domain_keywords = {
            'digital', 'technology', 'innovation', 'data', 'analytics', 'transformation',
            'agility', 'collaboration', 'leadership', 'strategy', 'culture', 'change',
            'growth', 'efficiency', 'automation', 'intelligence', 'customer', 'experience',
            'performance', 'excellence', 'optimization', 'scalability', 'competitive'
        }
        return word in domain_keywords
    
    def _extract_context_phrase(self, sentences: List) -> str:
        """Extract context phrase from first sentence."""
        if sentences:
            first_sentence = sentences[0].text
            # Extract first few words as context
            words = first_sentence.split()[:5]
            return " ".join(words) if words else "Digital leadership statement:"
        return "Digital leadership statement:"
    
    def _calculate_multi_factor_confidence(self, similarities: List[Tuple[int, str, float]], 
                                         cluster_stats: ClusterStatistics,
                                         adaptive_thresholds: AdaptiveThresholds) -> MultiFactorConfidence:
        """Calculate comprehensive multi-factor confidence assessment."""
        if not similarities:
            return MultiFactorConfidence(0, 0, 0, 0, 0, 0, 0, 0)
        
        primary_similarity = similarities[0][2]
        primary_cluster_id = similarities[0][0]
        
        # Base similarity score
        base_similarity = primary_similarity
        
        # Percentile rank among all similarities
        all_sims = [sim[2] for sim in similarities]
        percentile_rank = sum(1 for sim in all_sims if sim < primary_similarity) / len(all_sims)
        
        # Gap to second-best match
        gap_to_second = (primary_similarity - similarities[1][2]) if len(similarities) > 1 else 0.5
        
        # Cluster size factor (larger clusters are more reliable)
        cluster_size_factor = 1.0
        if cluster_stats.cluster_sizes:
            try:
                cluster_size = cluster_stats.cluster_sizes[primary_cluster_id]
                cluster_size_factor = min(cluster_size / self.min_cluster_size_for_reliability, 1.0)
            except (IndexError, KeyError):
                cluster_size_factor = 0.8
        
        # Cluster quality factor
        cluster_quality_factor = 1.0
        if cluster_stats.cluster_quality_scores:
            try:
                cluster_quality_factor = cluster_stats.cluster_quality_scores[primary_cluster_id]
            except (IndexError, KeyError):
                cluster_quality_factor = 0.7
        
        # Semantic boost (placeholder - will be filled in semantic analysis)
        semantic_boost = 0.0
        
        # Model compatibility penalty
        model_compatibility_penalty = 0.0
        
        # Calculate final confidence
        final_confidence = (
            base_similarity * 0.4 +
            percentile_rank * 0.2 +
            gap_to_second * 0.15 +
            cluster_size_factor * 0.1 +
            cluster_quality_factor * 0.1 +
            semantic_boost * 0.05
        ) - model_compatibility_penalty
        
        final_confidence = max(0.0, min(1.0, final_confidence))
        
        return MultiFactorConfidence(
            base_similarity=base_similarity,
            percentile_rank=percentile_rank,
            gap_to_second=gap_to_second,
            cluster_size_factor=cluster_size_factor,
            cluster_quality_factor=cluster_quality_factor,
            semantic_boost=semantic_boost,
            model_compatibility_penalty=model_compatibility_penalty,
            final_confidence=final_confidence
        )
    
    def _perform_semantic_analysis(self, sentence: str, domain_keywords: Dict[int, List[str]], 
                                 similarities: List[Tuple[int, str, float]]) -> Dict[int, SemanticAnalysis]:
        """Perform semantic analysis for each cluster match."""
        analyses = {}
        sentence_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', sentence.lower()))
        
        for cluster_id, archetype, similarity in similarities:
            cluster_keywords = set(domain_keywords.get(cluster_id, []))
            
            # Calculate keyword overlap
            matched_keywords = list(sentence_words & cluster_keywords)
            missing_keywords = list(cluster_keywords - sentence_words)
            
            keyword_overlap_score = len(matched_keywords) / max(len(cluster_keywords), 1)
            
            # Domain terminology presence
            domain_terminology_present = len(matched_keywords) > 0
            
            # Semantic coherence (simplified - based on keyword density)
            semantic_coherence_score = min(len(matched_keywords) / max(len(sentence_words), 1), 1.0)
            
            # Terminology boost
            terminology_boost = keyword_overlap_score * self.semantic_boost_factor
            
            analyses[cluster_id] = SemanticAnalysis(
                keyword_overlap_score=keyword_overlap_score,
                domain_terminology_present=domain_terminology_present,
                semantic_coherence_score=semantic_coherence_score,
                terminology_boost=terminology_boost,
                matched_keywords=matched_keywords,
                missing_keywords=missing_keywords
            )
        
        return analyses
    
    def _create_inference_matches(self, similarities: List[Tuple[int, str, float]],
                                semantic_analyses: Dict[int, SemanticAnalysis],
                                metadata: EnhancedReferenceMetadata,
                                multi_factor_confidence: MultiFactorConfidence) -> Tuple[InferenceMatch, List[InferenceMatch]]:
        """Create primary and alternative inference matches."""
        matches = []
        
        for i, (cluster_id, archetype, similarity) in enumerate(similarities):
            semantic_analysis = semantic_analyses.get(cluster_id, SemanticAnalysis(0, False, 0, 0, [], []))
            
            # Adjust confidence with semantic boost
            adjusted_confidence = similarity + semantic_analysis.terminology_boost
            adjusted_confidence = max(0.0, min(1.0, adjusted_confidence))
            
            # Determine confidence level
            confidence_level = self._determine_confidence_level(
                adjusted_confidence, metadata.adaptive_thresholds, i == 0, multi_factor_confidence
            )
            
            # Calculate reliability score
            reliability_score = self._calculate_reliability_score(
                cluster_id, metadata.cluster_statistics, adjusted_confidence
            )
            
            match = InferenceMatch(
                cluster_id=cluster_id,
                archetype=archetype,
                similarity_score=similarity,
                adjusted_confidence=adjusted_confidence,
                confidence_level=confidence_level,
                reliability_score=reliability_score,
                semantic_analysis=semantic_analysis
            )
            
            matches.append(match)
        
        # Split into primary and alternatives
        primary_match = matches[0] if matches else None
        alternative_matches = matches[1:self.max_alternative_matches] if len(matches) > 1 else []
        
        return primary_match, alternative_matches
    
    def _determine_confidence_level(self, confidence: float, thresholds: AdaptiveThresholds,
                                  is_primary: bool, multi_factor: MultiFactorConfidence) -> ConfidenceLevel:
        """Determine confidence level based on thresholds and context."""
        if multi_factor.base_similarity > thresholds.training_leakage_threshold:
            return ConfidenceLevel.TRAINING_MATCH
        
        if confidence >= thresholds.strong_threshold and multi_factor.gap_to_second > thresholds.ambiguity_gap_threshold:
            return ConfidenceLevel.STRONG
        elif confidence >= thresholds.weak_threshold:
            if is_primary and multi_factor.gap_to_second <= thresholds.ambiguity_gap_threshold:
                return ConfidenceLevel.AMBIGUOUS
            return ConfidenceLevel.WEAK
        else:
            return ConfidenceLevel.NO_MATCH
    
    def _calculate_reliability_score(self, cluster_id: int, cluster_stats: ClusterStatistics, 
                                   confidence: float) -> float:
        """Calculate reliability score for a match."""
        base_reliability = confidence
        
        # Adjust for cluster size
        if cluster_stats.cluster_sizes:
            try:
                cluster_size = cluster_stats.cluster_sizes[cluster_id]
                size_factor = min(cluster_size / self.min_cluster_size_for_reliability, 1.0)
                base_reliability *= (0.7 + 0.3 * size_factor)
            except (IndexError, KeyError):
                base_reliability *= 0.8
        
        # Adjust for cluster quality
        if cluster_stats.cluster_quality_scores:
            try:
                quality_factor = cluster_stats.cluster_quality_scores[cluster_id]
                base_reliability *= (0.7 + 0.3 * quality_factor)
            except (IndexError, KeyError):
                base_reliability *= 0.8
        
        return max(0.0, min(1.0, base_reliability))
    
    def _determine_classification_status(self, primary_match: InferenceMatch,
                                       alternative_matches: List[InferenceMatch],
                                       leakage_detected: bool,
                                       multi_factor: MultiFactorConfidence) -> ClassificationStatus:
        """Determine overall classification status."""
        if leakage_detected:
            return ClassificationStatus.TRAINING_LEAKAGE
        
        if not primary_match or primary_match.confidence_level == ConfidenceLevel.NO_MATCH:
            return ClassificationStatus.NO_MATCH
        
        if primary_match.confidence_level == ConfidenceLevel.STRONG:
            return ClassificationStatus.STRONG_MATCH
        
        if primary_match.confidence_level == ConfidenceLevel.AMBIGUOUS:
            return ClassificationStatus.AMBIGUOUS_MATCH
        
        # Check for multiple good matches
        good_alternatives = [m for m in alternative_matches 
                           if m.confidence_level in [ConfidenceLevel.STRONG, ConfidenceLevel.WEAK]
                           and m.adjusted_confidence > 0.7]
        
        if good_alternatives and multi_factor.gap_to_second < 0.1:
            return ClassificationStatus.MULTIPLE_GOOD_MATCHES
        
        return ClassificationStatus.WEAK_MATCH
    
    def _generate_warnings_and_recommendations(self, primary_match: InferenceMatch,
                                             alternative_matches: List[InferenceMatch],
                                             model_compatibility: float,
                                             leakage_detected: bool,
                                             thresholds: AdaptiveThresholds) -> Tuple[List[str], List[str]]:
        """Generate warnings and recommendations based on inference results."""
        warnings = []
        recommendations = []
        
        # Model compatibility warnings
        if model_compatibility < 0.9:
            warnings.append(f"Model mismatch detected (compatibility: {model_compatibility:.2f})")
            recommendations.append("Consider retraining reference database with current embedding model")
        
        # Training leakage warnings
        if leakage_detected:
            warnings.append("Potential training data leakage detected")
            recommendations.append("Verify if input sentence was part of training data")
        
        # Low confidence warnings
        if primary_match and primary_match.adjusted_confidence < thresholds.weak_threshold:
            warnings.append("Low confidence classification")
            recommendations.append("Consider manual review or additional context")
        
        # Ambiguity warnings
        if primary_match and primary_match.confidence_level == ConfidenceLevel.AMBIGUOUS:
            warnings.append("Ambiguous classification with multiple similar matches")
            recommendations.append("Review alternative matches for best fit")
        
        # Multiple good matches
        good_alternatives = [m for m in alternative_matches if m.adjusted_confidence > 0.7]
        if good_alternatives:
            warnings.append(f"Multiple good matches found ({len(good_alternatives)} alternatives)")
            recommendations.append("Consider reviewing top alternatives for context-specific fit")
        
        return warnings, recommendations
    
    def _validate_neighbors_parameters(self, n_neighbors: int, similarity_threshold: Optional[float]) -> None:
        """Validate nearest neighbors parameters."""
        if n_neighbors <= 0:
            raise ValueError("Number of neighbors must be positive")
        
        if n_neighbors > 100:
            raise ValueError("Number of neighbors cannot exceed 100")
        
        if similarity_threshold is not None:
            if not 0 <= similarity_threshold <= 1:
                raise ValueError("Similarity threshold must be between 0 and 1")
    
    def _process_search_results(self, search_results: List, n_neighbors: int, 
                              similarity_threshold: Optional[float]) -> List[Dict[str, Any]]:
        """Process Qdrant search results into neighbor format."""
        neighbors = []
        
        for i, result in enumerate(search_results):
            # Calculate similarity score (Qdrant returns distance, convert to similarity)
            similarity_score = 1 - result.score if hasattr(result, 'score') else 0.0
            
            # Apply similarity threshold if specified
            if similarity_threshold is not None and similarity_score < similarity_threshold:
                continue
                
            # Stop if we have enough neighbors
            if len(neighbors) >= n_neighbors:
                break
            
            neighbor = {
                'rank': len(neighbors) + 1,
                'sentence': result.payload.get('sentence', '').replace('Domain Logic example phrase: ', ''),
                'similarity_score': similarity_score,
                'point_id': str(result.id),
                'payload': result.payload
            }
            
            neighbors.append(neighbor)
        
        return neighbors
    
    def _extract_neighbor_dl_metadata(self, neighbors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract DL metadata from neighbor payloads."""
        neighbors_with_metadata = []
        
        for neighbor in neighbors:
            payload = neighbor.get('payload', {})
            
            # Extract DL metadata fields
            dl_metadata = {
                'dl_category': payload.get('dl_category'),
                'dl_subcategory': payload.get('dl_subcategory'), 
                'dl_archetype': payload.get('dl_archetype')
            }
            
            # Create enhanced neighbor object
            enhanced_neighbor = {
                'rank': neighbor['rank'],
                'sentence': neighbor['sentence'],
                'similarity_score': neighbor['similarity_score'],
                'point_id': neighbor['point_id'],
                **dl_metadata
            }
            
            neighbors_with_metadata.append(enhanced_neighbor)
        
        return neighbors_with_metadata
    
    def _analyze_neighbor_dominant_logic(self, neighbors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze dominant logic patterns across neighbors."""
        from collections import Counter
        
        # Extract DL elements (excluding None values)
        categories = [n['dl_category'] for n in neighbors if n['dl_category']]
        subcategories = [n['dl_subcategory'] for n in neighbors if n['dl_subcategory']]
        archetypes = [n['dl_archetype'] for n in neighbors if n['dl_archetype']]
        
        # Calculate most common elements
        category_counter = Counter(categories)
        subcategory_counter = Counter(subcategories)
        archetype_counter = Counter(archetypes)
        
        # Calculate confidence scores (percentage of neighbors with dominant element)
        total_neighbors = len(neighbors)
        
        dominant_logic = {}
        
        if category_counter.most_common(1):
            dominant_category, count = category_counter.most_common(1)[0]
            dominant_logic['dominant_category'] = dominant_category
            dominant_logic['category_confidence'] = count / total_neighbors
        else:
            dominant_logic['dominant_category'] = None
            dominant_logic['category_confidence'] = 0.0
        
        if subcategory_counter.most_common(1):
            dominant_subcategory, count = subcategory_counter.most_common(1)[0]
            dominant_logic['dominant_subcategory'] = dominant_subcategory
            dominant_logic['subcategory_confidence'] = count / total_neighbors
        else:
            dominant_logic['dominant_subcategory'] = None
            dominant_logic['subcategory_confidence'] = 0.0
        
        if archetype_counter.most_common(1):
            dominant_archetype, count = archetype_counter.most_common(1)[0]
            dominant_logic['dominant_archetype'] = dominant_archetype
            dominant_logic['archetype_confidence'] = count / total_neighbors
        else:
            dominant_logic['dominant_archetype'] = None
            dominant_logic['archetype_confidence'] = 0.0
        
        return dominant_logic
    
    def _calculate_neighbor_distribution_stats(self, neighbors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate distribution statistics for neighbor DL metadata."""
        from collections import Counter
        
        # Extract DL elements
        categories = [n['dl_category'] for n in neighbors if n['dl_category']]
        subcategories = [n['dl_subcategory'] for n in neighbors if n['dl_subcategory']]
        archetypes = [n['dl_archetype'] for n in neighbors if n['dl_archetype']]
        
        # Calculate distributions
        stats = {
            'category_distribution': dict(Counter(categories)),
            'subcategory_distribution': dict(Counter(subcategories)),
            'archetype_distribution': dict(Counter(archetypes)),
            'total_neighbors': len(neighbors),
            'neighbors_with_category': len(categories),
            'neighbors_with_subcategory': len(subcategories),
            'neighbors_with_archetype': len(archetypes),
            'unique_categories': len(set(categories)),
            'unique_subcategories': len(set(subcategories)),
            'unique_archetypes': len(set(archetypes)),
            'metadata_completeness': {
                'category_coverage': len(categories) / len(neighbors) if neighbors else 0,
                'subcategory_coverage': len(subcategories) / len(neighbors) if neighbors else 0,
                'archetype_coverage': len(archetypes) / len(neighbors) if neighbors else 0
            }
        }
        return stats
