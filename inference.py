#!/usr/bin/env python3
"""
Digital Leadership Inference CLI

A command-line interface for inferring Digital Leadership archetypes from text sentences
using the quantize_culture inference engine and reference database.

Usage:
    python inference.py --sentence "your text here" --collection collection_name [options]
    
    python inference.py -s "order is the key for success" -c extended_contextualized_collection
    python inference.py -s "fail fast learn faster" -c extended_contextualized_collection --format json
    python inference.py -s "innovation drives us forward" -c extended_contextualized_collection --verbose

Requirements:
    - OpenAI API key configured in environment (OPENAI_API_KEY)  
    - Qdrant vector database running and accessible
    - Reference collection populated with DL-labeled sentences

Output:
    By default, outputs human-readable classification results.
    With --format json, outputs structured JSON for programmatic use.
    With --verbose, includes additional debug information.

Author: Digital Leadership Assessment Pipeline
Version: 1.0
"""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "app"))

try:
    from app.core.inference_engine import EnhancedInferenceEngine
    from app.services.qdrant_client import QdrantService
    from app.services.openai_client import OpenAIService  
    from app.data.inference_models import EnhancedInferenceResult
except ImportError as e:
    print(f"Error importing required modules: {e}", file=sys.stderr)
    print("Make sure you're running from the project root directory.", file=sys.stderr)
    sys.exit(1)


class InferenceCLI:
    """Command-line interface for Digital Leadership inference."""
    
    def __init__(self):
        """Initialize the CLI with required services."""
        try:
            self.qdrant_service = QdrantService()
            self.openai_service = OpenAIService() 
            self.inference_engine = EnhancedInferenceEngine(self.openai_service, self.qdrant_service)
        except Exception as e:
            print(f"Error initializing services: {e}", file=sys.stderr)
            sys.exit(1)
    
    def infer_sentence(self, sentence: str, collection: str, verbose: bool = False) -> Dict[str, Any]:
        """
        Perform inference on a sentence and return structured results.
        
        Args:
            sentence: The input sentence to classify
            collection: Name of the Qdrant collection to use as reference
            verbose: Whether to include additional debug information
            
        Returns:
            Dictionary containing inference results
        """
        try:
            if verbose:
                print(f"Processing sentence: '{sentence}'", file=sys.stderr)
                print(f"Using collection: '{collection}'", file=sys.stderr)
            
            # Perform the inference
            result = self.inference_engine.infer_dl_archetype(sentence, collection)
            
            # Structure the output
            output = {
                "sentence": sentence,
                "collection": collection,
                "cluster_id": result.primary_match.cluster_id,
                "similarity_score": float(result.primary_match.similarity_score),
                "classification": result.primary_match.archetype,
                "confidence_level": result.primary_match.confidence_level.value,
                "classification_status": result.classification_status.value,
                "timestamp": result.processing_timestamp
            }
            
            if verbose:
                # Add additional debug information
                output["debug"] = {
                    "alternative_matches_count": len(result.alternative_matches),
                    "contextualized_sentence": result.contextualized_sentence,
                    "training_leakage_detected": result.training_leakage_detected,
                    "model_compatibility_score": result.model_compatibility_score,
                    "warnings": result.warnings,
                    "recommendations": result.recommendations
                }
                
                # Get ALL sentences from the matched cluster when verbose
                try:
                    all_sentences = self.qdrant_service.extract_data(collection)
                    cluster_sentences = [s for s in all_sentences if s.cluster_id == result.primary_match.cluster_id]
                    
                    if cluster_sentences:
                        output["cluster_members"] = [
                            {
                                "text": s.text,
                                "dl_category": getattr(s, 'dl_category', None),
                                "dl_subcategory": getattr(s, 'dl_subcategory', None), 
                                "dl_archetype": getattr(s, 'dl_archetype', None)
                            }
                            for s in cluster_sentences
                        ]
                        output["cluster_size"] = len(cluster_sentences)
                        
                        # Determine dominant logic from cluster members
                        dl_categories = [getattr(s, 'dl_category', None) for s in cluster_sentences if getattr(s, 'dl_category', None)]
                        dl_subcategories = [getattr(s, 'dl_subcategory', None) for s in cluster_sentences if getattr(s, 'dl_subcategory', None)]
                        dl_archetypes = [getattr(s, 'dl_archetype', None) for s in cluster_sentences if getattr(s, 'dl_archetype', None)]
                        
                        # Find most common dominant logic elements
                        from collections import Counter
                        
                        output["dominant_logic"] = {
                            "most_common_category": Counter(dl_categories).most_common(1)[0] if dl_categories else None,
                            "most_common_subcategory": Counter(dl_subcategories).most_common(1)[0] if dl_subcategories else None,
                            "most_common_archetype": Counter(dl_archetypes).most_common(1)[0] if dl_archetypes else None,
                            "category_distribution": dict(Counter(dl_categories)) if dl_categories else {},
                            "subcategory_distribution": dict(Counter(dl_subcategories)) if dl_subcategories else {},
                            "archetype_distribution": dict(Counter(dl_archetypes)) if dl_archetypes else {}
                        }
                        
                except Exception as e:
                    if verbose:
                        print(f"Warning: Could not retrieve cluster members: {e}", file=sys.stderr)
            
            return output
            
        except Exception as e:
            error_output = {
                "error": str(e),
                "sentence": sentence,
                "collection": collection,
                "success": False
            }
            if verbose:
                import traceback
                error_output["traceback"] = traceback.format_exc()
            return error_output
    
    def format_human_readable(self, result: Dict[str, Any]) -> str:
        """Format inference results in human-readable format."""
        if "error" in result:
            return f"❌ Error: {result['error']}"
        
        output = []
        output.append(f"📝 Sentence: \"{result['sentence']}\"")
        output.append(f"🎯 Digital Leadership Classification: {result['classification']}")
        output.append(f"📊 Cluster ID: {result['cluster_id']}")
        output.append(f"🔍 Similarity Score: {result['similarity_score']:.4f}")
        output.append(f"✅ Confidence Level: {result['confidence_level']}")
        
        # Show dominant logic if available
        if "dominant_logic" in result:
            dl = result["dominant_logic"]
            output.append(f"\n🧠 DOMINANT LOGIC ANALYSIS:")
            
            if dl["most_common_category"]:
                category, count = dl["most_common_category"]
                output.append(f"   📂 Primary Category: {category} ({count}/{result.get('cluster_size', 0)} sentences)")
            
            if dl["most_common_subcategory"]:
                subcategory, count = dl["most_common_subcategory"]
                output.append(f"   📋 Primary Subcategory: {subcategory} ({count}/{result.get('cluster_size', 0)} sentences)")
                
            if dl["most_common_archetype"]:
                archetype, count = dl["most_common_archetype"]
                output.append(f"   🎭 Primary Archetype: {archetype} ({count}/{result.get('cluster_size', 0)} sentences)")
        
        if "cluster_members" in result:
            output.append(f"\n📚 ALL CLUSTER MEMBERS ({result.get('cluster_size', 0)} total):")
            for i, member in enumerate(result["cluster_members"], 1):
                dl_info = ""
                if member.get('dl_category') or member.get('dl_subcategory') or member.get('dl_archetype'):
                    dl_parts = [
                        member.get('dl_category', ''),
                        member.get('dl_subcategory', ''),
                        member.get('dl_archetype', '')
                    ]
                    dl_info = f" [{' → '.join(filter(None, dl_parts))}]"
                output.append(f"   {i:2d}. {member['text']}{dl_info}")
                
        elif "sample_cluster_sentences" in result:
            output.append(f"\n📚 Sample sentences from this cluster:")
            for i, sentence in enumerate(result["sample_cluster_sentences"], 1):
                output.append(f"   {i}. {sentence}")
        
        return "\n".join(output)


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Infer Digital Leadership archetype from text using vector similarity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -s "order is the key for success" -c extended_contextualized_collection
  %(prog)s -s "fail fast learn faster" -c extended_contextualized_collection --format json
  %(prog)s -s "innovation drives us" -c extended_contextualized_collection --verbose
        """
    )
    
    parser.add_argument(
        "-s", "--sentence",
        required=True,
        help="The sentence to classify for Digital Leadership archetype"
    )
    
    parser.add_argument(
        "-c", "--collection", 
        required=True,
        help="The Qdrant collection name to use as reference database"
    )
    
    parser.add_argument(
        "--format",
        choices=["json", "human"],
        default="human",
        help="Output format (default: human)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Include additional debug information in output"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="Digital Leadership Inference CLI v1.0"
    )
    
    return parser


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Validate environment
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set", file=sys.stderr)
        print("Please set your OpenAI API key: export OPENAI_API_KEY=your_key_here", file=sys.stderr)
        sys.exit(1)
    
    # Initialize CLI and perform inference
    try:
        cli = InferenceCLI()
        result = cli.infer_sentence(args.sentence, args.collection, args.verbose)
        
        # Output results in requested format
        if args.format == "json":
            print(json.dumps(result, indent=2))
        else:
            print(cli.format_human_readable(result))
            
        # Exit with error code if inference failed
        if "error" in result:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"❌ Unexpected error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
