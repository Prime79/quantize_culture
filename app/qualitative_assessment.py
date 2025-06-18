#!/usr/bin/env python3
"""
Qualitative (Semantic/Cultural) clustering assessment for company culture analysis.

This module provides semantic and cultural quality metrics to complement
the quantitative (mathematical/statistical) clustering metrics.

QUANTITATIVE MEASURES (mathematical/statistical):
- Silhouette score, Davies-Bouldin index
- Noise percentage, cluster count
- UMAP/HDBSCAN parameter optimization

QUALITATIVE MEASURES (semantic/cultural):
- Semantic coherence within clusters
- Cultural dimension alignment  
- Business interpretability
- Actionable insights quality
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import openai
from dotenv import load_dotenv
import os
import json
from sklearn.metrics.pairwise import cosine_similarity
import asyncio
import warnings
warnings.filterwarnings('ignore')

load_dotenv()

class QualitativeClusteringAssessment:
    """
    Assess clustering quality from semantic and cultural perspective.
    
    This class evaluates QUALITATIVE (semantic/cultural) aspects of clustering,
    as opposed to QUANTITATIVE (mathematical/statistical) measures.
    """
    
    def __init__(self):
        """Initialize qualitative assessment with cultural frameworks."""
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Organizational culture dimensions based on research (Cameron & Quinn, Hofstede, etc.)
        self.cultural_dimensions = {
            'performance_excellence': {
                'keywords': ['results', 'achieve', 'goals', 'deliver', 'excellence', 
                           'success', 'performance', 'metrics', 'targets', 'outcome'],
                'description': 'Focus on achieving results and high performance'
            },
            'innovation_creativity': {
                'keywords': ['innovation', 'creative', 'new', 'ideas', 'change', 
                           'improve', 'think', 'experiment', 'breakthrough', 'disrupt'],
                'description': 'Emphasis on creativity and innovative thinking'
            },
            'collaboration_teamwork': {
                'keywords': ['team', 'collaborate', 'together', 'support', 'help', 
                           'share', 'unity', 'cooperation', 'partnership', 'collective'],
                'description': 'Valuing teamwork and collaborative approaches'
            },
            'integrity_ethics': {
                'keywords': ['integrity', 'honest', 'ethical', 'trust', 'respect', 
                           'values', 'principle', 'moral', 'fair', 'transparent'],
                'description': 'Strong emphasis on ethical behavior and values'
            },
            'customer_centricity': {
                'keywords': ['customer', 'client', 'service', 'satisfaction', 
                           'experience', 'relationship', 'value', 'quality', 'focus'],
                'description': 'Putting customers at the center of decisions'
            },
            'leadership_empowerment': {
                'keywords': ['leadership', 'lead', 'empower', 'guide', 'inspire', 
                           'vision', 'responsibility', 'decision', 'autonomy', 'ownership'],
                'description': 'Strong leadership and employee empowerment'
            },
            'learning_growth': {
                'keywords': ['learn', 'grow', 'develop', 'skill', 'knowledge', 
                           'education', 'training', 'improvement', 'career', 'mastery'],
                'description': 'Commitment to continuous learning and development'
            },
            'work_life_integration': {
                'keywords': ['balance', 'flexible', 'family', 'personal', 'wellbeing', 
                           'health', 'time', 'life', 'happiness', 'wellness'],
                'description': 'Supporting work-life balance and employee wellbeing'
            },
            'adaptability_agility': {
                'keywords': ['adapt', 'agile', 'flexible', 'responsive', 'change', 
                           'quick', 'nimble', 'evolve', 'pivot', 'adjust'],
                'description': 'Ability to adapt quickly to changing circumstances'
            },
            'quality_craftsmanship': {
                'keywords': ['quality', 'excellence', 'detail', 'precision', 'craft', 
                           'standard', 'meticulous', 'thorough', 'careful', 'pride'],
                'description': 'Dedication to quality and attention to detail'
            }
        }
    
    def assess_cluster_semantic_coherence(self, sentences: List[str]) -> float:
        """
        QUALITATIVE MEASURE: Assess semantic coherence within cluster.
        
        Evaluates how semantically similar sentences are within a cluster
        using embedding cosine similarity - a qualitative (semantic) measure
        as opposed to quantitative (mathematical) clustering metrics.
        
        Args:
            sentences: List of sentences in the cluster
            
        Returns:
            Semantic coherence score (0-1, higher = more coherent)
        """
        if len(sentences) < 2:
            return 0.0
        
        try:
            # Get embeddings for all sentences
            embeddings = []
            for sentence in sentences[:10]:  # Limit for API efficiency
                response = self.client.embeddings.create(
                    input=sentence,
                    model="text-embedding-3-small"
                )
                embeddings.append(response.data[0].embedding)
            
            # Calculate pairwise similarities
            similarities = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                    similarities.append(sim)
            
            return float(np.mean(similarities))
            
        except Exception as e:
            print(f"Warning: Coherence calculation failed: {e}")
            return 0.5  # Default moderate score
    
    def assess_cultural_alignment(self, sentences: List[str]) -> Dict:
        """
        QUALITATIVE MEASURE: Assess cultural dimension alignment.
        
        Evaluates how well cluster aligns with organizational culture dimensions.
        This is a qualitative (cultural/semantic) assessment, complementing
        quantitative (mathematical/statistical) clustering metrics.
        
        Args:
            sentences: List of sentences in the cluster
            
        Returns:
            Dictionary with cultural alignment scores and dominant dimension
        """
        dimension_scores = {}
        sentence_text = ' '.join(sentences).lower()
        
        for dimension, info in self.cultural_dimensions.items():
            keywords = info['keywords']
            
            # Count keyword matches with context
            matches = 0
            total_keywords = len(keywords)
            
            for keyword in keywords:
                if keyword in sentence_text:
                    # Weight by frequency but cap to avoid keyword stuffing bias
                    freq = min(sentence_text.count(keyword), 3)
                    matches += freq
            
            # Normalize by total keywords and sentence count
            score = matches / (total_keywords * len(sentences))
            dimension_scores[dimension] = min(score, 1.0)  # Cap at 1.0
        
        # Find dominant dimension
        max_dimension = max(dimension_scores.items(), key=lambda x: x[1])
        
        # Calculate alignment strength (how focused vs scattered)
        total_score = sum(dimension_scores.values())
        alignment_strength = max_dimension[1] / total_score if total_score > 0 else 0
        
        return {
            'dimension_scores': dimension_scores,
            'dominant_dimension': max_dimension[0],
            'dominant_score': max_dimension[1],
            'alignment_strength': alignment_strength,
            'cultural_focus': max_dimension[1] * alignment_strength
        }
    
    def assess_interpretability(self, sentences: List[str], max_sentences: int = 5) -> Dict:
        """
        QUALITATIVE MEASURE: Assess business interpretability using LLM.
        
        Uses LLM to evaluate whether the cluster represents a coherent business theme
        and provides actionable insights. This is a qualitative (semantic/business)
        assessment complementing quantitative (mathematical) clustering metrics.
        
        Args:
            sentences: List of sentences in the cluster
            max_sentences: Maximum sentences to include in LLM prompt
            
        Returns:
            Dictionary with business interpretability metrics
        """
        if len(sentences) > max_sentences:
            sentences = sentences[:max_sentences]
        
        prompt = f"""
        Analyze these company culture statements and assess their coherence:
        
        {chr(10).join(f"- {sentence}" for sentence in sentences)}
        
        Please provide:
        1. Coherence score (0-1): How well do these statements represent a single theme?
        2. Theme name (2-4 words): What cultural theme do they represent?
        3. Business value (0-1): How actionable/valuable is this insight for management?
        
        Respond in JSON format:
        {{
            "coherence_score": 0.8,
            "theme_name": "Customer Focus Excellence",
            "business_value": 0.9,
            "explanation": "Brief explanation of the theme"
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.1
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            if result_text.startswith("```json"):
                result_text = result_text.replace("```json", "").replace("```", "").strip()
            
            result = json.loads(result_text)
            
            return {
                'llm_coherence': float(result.get('coherence_score', 0.5)),
                'theme_name': result.get('theme_name', 'Unknown Theme'),
                'business_value': float(result.get('business_value', 0.5)),
                'explanation': result.get('explanation', 'No explanation provided')
            }
            
        except Exception as e:
            print(f"Warning: LLM interpretability assessment failed: {e}")
            return {
                'llm_coherence': 0.5,
                'theme_name': 'Analysis Failed',
                'business_value': 0.3,
                'explanation': f'Assessment failed: {str(e)}'
            }
    
    def assess_single_cluster_quality(self, sentences: List[str], cluster_id: str) -> Dict:
        """
        QUALITATIVE MEASURE: Comprehensive qualitative assessment of a single cluster.
        
        Combines semantic coherence, cultural alignment, and business interpretability
        into an overall qualitative quality score. This complements quantitative
        (mathematical/statistical) clustering metrics.
        
        Args:
            sentences: List of sentences in the cluster
            cluster_id: Identifier for the cluster
            
        Returns:
            Complete qualitative assessment with semantic and cultural metrics
        """
        if len(sentences) < 2:
            return {
                'cluster_id': cluster_id,
                'cluster_size': len(sentences),
                'semantic_coherence': 0.0,
                'cultural_alignment': {'cultural_focus': 0.0, 'dominant_dimension': 'none'},
                'interpretability': {'business_value': 0.0, 'theme_name': 'Too Small'},
                'overall_qualitative_score': 0.0,
                'quality_grade': 'F',
                'sample_sentences': sentences
            }
        
        print(f"  Assessing cluster {cluster_id} ({len(sentences)} sentences)...")
        
        # 1. Semantic coherence
        coherence = self.assess_cluster_semantic_coherence(sentences)
        
        # 2. Cultural alignment
        cultural = self.assess_cultural_alignment(sentences)
        
        # 3. Business interpretability
        interpretability = self.assess_interpretability(sentences)
        
        # 4. Calculate overall qualitative score
        overall_score = (
            coherence * 0.3 +
            cultural['cultural_focus'] * 0.3 +
            interpretability['business_value'] * 0.4
        )
        
        # 5. Assign quality grade
        if overall_score >= 0.8:
            grade = 'A'
        elif overall_score >= 0.6:
            grade = 'B'
        elif overall_score >= 0.4:
            grade = 'C'
        elif overall_score >= 0.2:
            grade = 'D'
        else:
            grade = 'F'
        
        return {
            'cluster_id': cluster_id,
            'cluster_size': len(sentences),
            'semantic_coherence': round(coherence, 3),
            'cultural_alignment': {
                'dominant_dimension': cultural['dominant_dimension'],
                'dominant_score': round(cultural['dominant_score'], 3),
                'cultural_focus': round(cultural['cultural_focus'], 3),
                'alignment_strength': round(cultural['alignment_strength'], 3)
            },
            'interpretability': {
                'llm_coherence': round(interpretability['llm_coherence'], 3),
                'theme_name': interpretability['theme_name'],
                'business_value': round(interpretability['business_value'], 3),
                'explanation': interpretability['explanation']
            },
            'overall_qualitative_score': round(overall_score, 3),
            'quality_grade': grade,
            'sample_sentences': sentences[:3]  # First 3 as examples
        }
    
    def assess_full_clustering_qualitative(self, clusters_dict: Dict) -> Dict:
        """
        QUALITATIVE MEASURE: Assess qualitative quality of entire clustering result.
        
        Evaluates semantic and cultural aspects across all clusters, providing
        qualitative (semantic/business/cultural) metrics to complement
        quantitative (mathematical/statistical) clustering assessment.
        
        Args:
            clusters_dict: Dictionary mapping cluster_id -> list of sentences
            
        Returns:
            Comprehensive qualitative assessment across all clusters
        """
        print("🎨 QUALITATIVE CLUSTERING ASSESSMENT")
        print("=" * 50)
        
        cluster_assessments = {}
        quality_scores = []
        business_values = []
        cultural_coverage = defaultdict(int)
        
        # Assess each cluster
        for cluster_id, sentences in clusters_dict.items():
            if cluster_id == 'noise' or cluster_id == -1:
                continue
            
            assessment = self.assess_single_cluster_quality(sentences, cluster_id)
            cluster_assessments[cluster_id] = assessment
            
            # Collect aggregate metrics
            quality_scores.append(assessment['overall_qualitative_score'])
            business_values.append(assessment['interpretability']['business_value'])
            
            # Track cultural dimension coverage
            dominant_dim = assessment['cultural_alignment']['dominant_dimension']
            cultural_coverage[dominant_dim] += 1
        
        # Calculate aggregate qualitative metrics
        total_statements = sum(len(sentences) for cluster_id, sentences in clusters_dict.items() if cluster_id != -1)
        noise_count = len(clusters_dict.get(-1, []))
        
        aggregate_assessment = {
            'total_clusters_assessed': len(cluster_assessments),
            'average_qualitative_score': round(np.mean(quality_scores) if quality_scores else 0, 3),
            'average_business_value': round(np.mean(business_values) if business_values else 0, 3),
            'total_statements': total_statements,
            'noise_statements': noise_count,
            'cultural_dimensions_covered': len(cultural_coverage),
            'cultural_distribution': dict(cultural_coverage),
            'quality_grade_distribution': self._calculate_grade_distribution(cluster_assessments),
            'top_quality_clusters': self._get_top_clusters(cluster_assessments, 5),
            'improvement_opportunities': self._identify_improvement_opportunities(cluster_assessments),
            'cluster_details': cluster_assessments
        }
        
        return aggregate_assessment
    
    def _calculate_grade_distribution(self, assessments: Dict) -> Dict:
        """Calculate distribution of quality grades."""
        grades = [assessment['quality_grade'] for assessment in assessments.values()]
        grade_counts = {grade: grades.count(grade) for grade in ['A', 'B', 'C', 'D', 'F']}
        return grade_counts
    
    def _get_top_clusters(self, assessments: Dict, n: int = 5) -> List[Dict]:
        """Get top N clusters by qualitative score."""
        sorted_clusters = sorted(
            assessments.items(),
            key=lambda x: x[1]['overall_qualitative_score'],
            reverse=True
        )
        
        return [
            {
                'cluster_id': cluster_id,
                'score': assessment['overall_qualitative_score'],
                'theme': assessment['interpretability']['theme_name'],
                'size': assessment['cluster_size']
            }
            for cluster_id, assessment in sorted_clusters[:n]
        ]
    
    def _identify_improvement_opportunities(self, assessments: Dict) -> List[str]:
        """Identify areas for clustering improvement."""
        opportunities = []
        
        # Check for low-quality clusters
        low_quality = [a for a in assessments.values() if a['overall_qualitative_score'] < 0.4]
        if len(low_quality) > len(assessments) * 0.3:
            opportunities.append("High proportion of low-quality clusters - consider adjusting parameters")
        
        # Check for cultural dimension gaps
        covered_dimensions = set(a['cultural_alignment']['dominant_dimension'] 
                               for a in assessments.values())
        if len(covered_dimensions) < 5:
            opportunities.append("Limited cultural dimension coverage - data may be biased")
        
        # Check for very small clusters
        tiny_clusters = [a for a in assessments.values() if a['cluster_size'] < 3]
        if len(tiny_clusters) > len(assessments) * 0.4:
            opportunities.append("Many tiny clusters - consider increasing min_cluster_size")
        
        # Check for low business value
        low_value = [a for a in assessments.values() 
                    if a['interpretability']['business_value'] < 0.5]
        if len(low_value) > len(assessments) * 0.4:
            opportunities.append("Low business value clusters - review cultural relevance")
        
        return opportunities if opportunities else ["Clustering quality looks good!"]

def run_qualitative_assessment(collection_name: str = "company_culture_embeddings") -> Dict:
    """
    Run complete QUALITATIVE (semantic/cultural) assessment on clustering results.
    
    This function evaluates the qualitative aspects of clustering results,
    focusing on semantic coherence, cultural alignment, and business interpretability.
    It complements quantitative (mathematical/statistical) clustering metrics.
    
    Args:
        collection_name: Name of the Qdrant collection to assess
        
    Returns:
        Comprehensive qualitative assessment results with semantic and cultural metrics
    """
    # Import here to avoid circular imports
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from extract import DataExtractorAnalyzer
    
    print("🎨 Starting Qualitative Clustering Assessment")
    print("=" * 60)
    
    # Extract clustering data
    analyzer = DataExtractorAnalyzer(collection_name)
    df = analyzer.extract_data()
    
    if df is None or df.empty:
        print("❌ No data found in collection")
        return {}
    
    # Group sentences by cluster
    clusters = defaultdict(list)
    for _, row in df.iterrows():
        cluster_id = row.get('cluster', -1)
        sentence = row.get('sentence', '')
        if sentence:
            clusters[cluster_id].append(sentence)
    
    print(f"📊 Found {len(clusters)} clusters with {len(df)} total statements")
    
    # Run qualitative assessment
    assessor = QualitativeClusteringAssessment()
    results = assessor.assess_full_clustering_qualitative(dict(clusters))
    
    # Print summary
    print(f"\n📈 QUALITATIVE ASSESSMENT SUMMARY")
    print(f"   Average Quality Score: {results['average_qualitative_score']:.3f}")
    print(f"   Average Business Value: {results['average_business_value']:.3f}")
    print(f"   Cultural Dimensions Covered: {results['cultural_dimensions_covered']}")
    print(f"   Quality Grade Distribution: {results['quality_grade_distribution']}")
    
    print(f"\n🏆 TOP QUALITY CLUSTERS:")
    for cluster in results['top_quality_clusters']:
        print(f"   {cluster['cluster_id']}: {cluster['theme']} (Score: {cluster['score']:.3f})")
    
    print(f"\n💡 IMPROVEMENT OPPORTUNITIES:")
    for opportunity in results['improvement_opportunities']:
        print(f"   • {opportunity}")
    
    return results

if __name__ == "__main__":
    results = run_qualitative_assessment()
