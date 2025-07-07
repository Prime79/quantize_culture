#!/usr/bin/env python3
"""
Improved LightGBM training pipeline for dominant logic classification.
Addresses class imbalance and small dataset issues with better preprocessing.
"""

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict
import time
from pathlib import Path
import joblib
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

def extract_and_preprocess_data(collection_name: str = "target_test") -> Tuple[np.ndarray, np.ndarray, List[str], Dict]:
    """
    Step 1: Extract and preprocess data with intelligent class filtering.
    
    Args:
        collection_name: Name of the Qdrant collection
        
    Returns:
        tuple: (umap_embeddings, dominant_logics, passages, metadata)
    """
    print(f"🔍 Step 1: Extracting and preprocessing data from '{collection_name}'...")
    
    # Connect to Qdrant
    client = QdrantClient(host="localhost", port=6333)
    
    # Get all points with UMAP embeddings and non-empty dominant logic
    points, _ = client.scroll(
        collection_name=collection_name,
        limit=10000,
        with_payload=True,
        with_vectors=False  # We'll use UMAP from payload
    )
    
    print(f"📊 Retrieved {len(points)} total points from Qdrant")
    
    # Extract data and clean dominant logic labels
    umap_embeddings = []
    dominant_logics = []
    passages = []
    
    for point in points:
        # Get UMAP embedding from payload
        umap_vector = point.payload.get('umap_embedding')
        dominant_logic = point.payload.get('dominant_logic', '').strip()
        
        # Clean and normalize dominant logic labels
        if umap_vector and dominant_logic:
            # Remove parentheses and normalize
            cleaned_logic = dominant_logic.replace('(', '').replace(')', '').strip()
            
            # Skip very ambiguous or unclear labels
            skip_labels = ['NOT YET IDENTIFIED', 'UNK', 'UNKNOWN', '', 'NONE', 'NULL']
            if cleaned_logic.upper() not in skip_labels and len(cleaned_logic) > 2:
                umap_embeddings.append(umap_vector)
                dominant_logics.append(cleaned_logic)
                passages.append(point.payload.get('passage', ''))
    
    umap_embeddings = np.array(umap_embeddings)
    
    print(f"✅ Filtered to {len(umap_embeddings)} records with valid dominant logic")
    print(f"📈 UMAP embeddings shape: {umap_embeddings.shape}")
    
    # Analyze class distribution
    class_counts = Counter(dominant_logics)
    print(f"\n📊 Raw class distribution:")
    for logic, count in class_counts.most_common():
        print(f"   • {logic}: {count} samples")
    
    return umap_embeddings, np.array(dominant_logics), passages, {
        'class_counts': class_counts,
        'total_samples': len(umap_embeddings)
    }

def intelligent_class_filtering(X: np.ndarray, y: np.ndarray, passages: List[str], 
                               min_samples_per_class: int = 3) -> Tuple[np.ndarray, np.ndarray, List[str], Dict]:
    """
    Step 2: Intelligently filter classes and potentially merge similar categories.
    
    Args:
        X: Feature vectors
        y: Labels
        passages: Corresponding passages
        min_samples_per_class: Minimum samples required per class
        
    Returns:
        Filtered data and metadata
    """
    print(f"\n🔄 Step 2: Intelligent class filtering and merging...")
    print(f"   Minimum samples per class: {min_samples_per_class}")
    
    # Count classes
    class_counts = Counter(y)
    print(f"📊 Original class distribution:")
    for logic, count in class_counts.most_common():
        print(f"   • {logic}: {count} samples")
    
    # Strategy 1: Group related categories
    def merge_similar_categories(labels):
        """Merge conceptually similar categories to increase sample sizes."""
        merged_labels = []
        for label in labels:
            label_upper = label.upper()
            
            # Group financial performance related
            if 'FINANCIAL PERFORMANCE' in label_upper:
                merged_labels.append('FINANCIAL_PERFORMANCE')
            # Group certainty related
            elif 'CERTAINTY' in label_upper and 'FINANCIAL' not in label_upper:
                merged_labels.append('CERTAINTY')
            # Group rules related
            elif 'RULES' in label_upper:
                merged_labels.append('RULES')
            # Group knowledge related
            elif 'KNOWLEDGE' in label_upper:
                merged_labels.append('KNOWLEDGE')
            # Keep others as is if they're frequent enough
            elif label_upper in ['ENTREPRENEUR', 'EXPERIMENT AND ITERATE']:
                merged_labels.append(label_upper)
            else:
                # Group less frequent categories as 'OTHER'
                merged_labels.append('OTHER')
        
        return merged_labels
    
    # Apply merging strategy
    y_merged = merge_similar_categories(y)
    
    # Check new distribution
    merged_counts = Counter(y_merged)
    print(f"\n📊 After merging similar categories:")
    for logic, count in merged_counts.most_common():
        print(f"   • {logic}: {count} samples")
    
    # Filter out classes with too few samples
    valid_classes = [cls for cls, count in merged_counts.items() if count >= min_samples_per_class]
    
    # Filter data
    mask = np.array([label in valid_classes for label in y_merged])
    X_filtered = X[mask]
    y_filtered = np.array(y_merged)[mask]
    passages_filtered = [passages[i] for i in range(len(passages)) if mask[i]]
    
    final_counts = Counter(y_filtered)
    print(f"\n✅ Final class distribution after filtering:")
    for logic, count in final_counts.most_common():
        print(f"   • {logic}: {count} samples")
    
    print(f"📊 Summary:")
    print(f"   • Original classes: {len(class_counts)}")
    print(f"   • After merging: {len(merged_counts)}")
    print(f"   • Final classes: {len(final_counts)}")
    print(f"   • Retained samples: {len(X_filtered)}/{len(X)} ({len(X_filtered)/len(X)*100:.1f}%)")
    
    return X_filtered, y_filtered, passages_filtered, {
        'original_classes': len(class_counts),
        'merged_classes': len(merged_counts),
        'final_classes': len(final_counts),
        'retention_rate': len(X_filtered)/len(X)
    }

def train_improved_lightgbm(X_train: np.ndarray, y_train: np.ndarray, 
                           X_test: np.ndarray, y_test: np.ndarray,
                           label_encoder: LabelEncoder) -> lgb.LGBMClassifier:
    """
    Step 3: Train LightGBM with class balancing and optimized parameters.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data  
        label_encoder: Fitted label encoder
        
    Returns:
        Trained LightGBM model
    """
    print(f"\n🚀 Step 3: Training improved LightGBM model...")
    
    # Calculate class weights to handle imbalance
    unique_classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_train)
    class_weight_dict = dict(zip(unique_classes, class_weights))
    
    print(f"⚖️  Class weights for balancing:")
    for cls, weight in class_weight_dict.items():
        cls_name = label_encoder.inverse_transform([cls])[0]
        print(f"   • {cls_name}: {weight:.3f}")
    
    # Optimized LightGBM parameters for small datasets
    model = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=len(unique_classes),
        boosting_type='gbdt',
        num_leaves=31,  # Smaller for small dataset
        max_depth=6,
        learning_rate=0.05,  # Lower learning rate
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        min_child_samples=5,  # Adjusted for small dataset
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_estimators=200,  # More trees with lower learning rate
        class_weight='balanced',  # Built-in class balancing
        importance_type='gain',
        force_col_wise=True,
        verbose=-1  # Suppress training output
    )
    
    print(f"📋 Model parameters:")
    key_params = ['n_estimators', 'max_depth', 'learning_rate', 'num_leaves', 'min_child_samples']
    for param in key_params:
        print(f"   • {param}: {getattr(model, param)}")
    
    # Train model
    print(f"⏳ Training model...")
    start_time = time.time()
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='multi_logloss',
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]  # Silent training
    )
    
    end_time = time.time()
    print(f"✅ Training completed in {end_time - start_time:.2f} seconds")
    print(f"🌳 Best iteration: {model.best_iteration_}")
    
    return model

def comprehensive_evaluation(model: lgb.LGBMClassifier, X_test: np.ndarray, y_test: np.ndarray,
                           label_encoder: LabelEncoder, passages_test: List[str]) -> Dict:
    """
    Step 4: Comprehensive model evaluation with detailed metrics.
    """
    print(f"\n📊 Step 4: Comprehensive model evaluation...")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Convert to original labels
    y_test_labels = label_encoder.inverse_transform(y_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred)
    
    # Basic metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"🎯 Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Classification report
    print(f"\n📈 Classification Report:")
    print("=" * 60)
    report = classification_report(
        y_test_labels, y_pred_labels, 
        output_dict=False, zero_division=0
    )
    print(report)
    
    # Feature importance
    feature_importance = model.feature_importances_
    print(f"\n🌟 Feature Importance (UMAP dimensions):")
    print("=" * 50)
    for i, importance in enumerate(feature_importance):
        print(f"   • UMAP_dim_{i}: {importance:.1f}")
    
    # Sample predictions with confidence
    print(f"\n🔍 Sample Predictions with Confidence:")
    print("=" * 50)
    for i in range(min(5, len(X_test))):
        true_label = y_test_labels[i]
        pred_label = y_pred_labels[i]
        confidence = np.max(y_pred_proba[i])
        match = "✅" if true_label == pred_label else "❌"
        
        print(f"   Sample {i+1}:")
        print(f"     True: {true_label}")
        print(f"     Pred: {pred_label} (confidence: {confidence:.3f})")
        print(f"     Match: {match}")
        print(f"     Passage: \"{passages_test[i][:80]}...\"")
        print()
    
    return {
        'accuracy': accuracy,
        'y_true': y_test_labels,
        'y_pred': y_pred_labels,
        'y_pred_proba': y_pred_proba,
        'feature_importance': feature_importance
    }

def cross_validation_analysis(X: np.ndarray, y: np.ndarray, label_encoder: LabelEncoder) -> None:
    """
    Perform cross-validation analysis to assess model stability.
    """
    print(f"\n📊 Cross-Validation Analysis...")
    
    # Create base model
    model = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=len(label_encoder.classes_),
        max_depth=6,
        learning_rate=0.05,
        n_estimators=100,
        class_weight='balanced',
        random_state=42,
        verbose=-1
    )
    
    # 5-fold stratified cross-validation
    cv_scores = cross_val_score(
        model, X, y, 
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='accuracy'
    )
    
    print(f"🎯 Cross-Validation Results:")
    print(f"   • Mean accuracy: {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})")
    print(f"   • Individual folds: {[f'{score:.3f}' for score in cv_scores]}")

def main():
    """Main function to orchestrate the improved ML pipeline."""
    print("🎯 Improved LightGBM Training Pipeline for Dominant Logic Classification")
    print("=" * 80)
    
    try:
        # Step 1: Extract and preprocess data
        X, y, passages, metadata = extract_and_preprocess_data()
        
        if len(X) < 20:
            print("❌ Insufficient data for reliable training")
            return
        
        # Step 2: Intelligent class filtering and merging
        X_filtered, y_filtered, passages_filtered, filter_metadata = intelligent_class_filtering(
            X, y, passages, min_samples_per_class=3
        )
        
        if len(X_filtered) < 15:
            print("❌ Insufficient data after filtering")
            return
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y_filtered)
        
        print(f"\n📊 Final dataset summary:")
        print(f"   • Total samples: {len(X_filtered)}")
        print(f"   • Features: {X_filtered.shape[1]}D UMAP vectors")
        print(f"   • Classes: {len(label_encoder.classes_)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_filtered, y_encoded, 
            test_size=0.25, 
            random_state=42,
            stratify=y_encoded
        )
        
        # Split passages for evaluation
        passages_array = np.array(passages_filtered)
        _, passages_test, _, _ = train_test_split(
            passages_array, y_encoded, 
            test_size=0.25, 
            random_state=42,
            stratify=y_encoded
        )
        
        print(f"✅ Data split: {len(X_train)} train, {len(X_test)} test")
        
        # Step 3: Train improved model
        model = train_improved_lightgbm(X_train, y_train, X_test, y_test, label_encoder)
        
        # Step 4: Comprehensive evaluation
        results = comprehensive_evaluation(model, X_test, y_test, label_encoder, passages_test.tolist())
        
        # Cross-validation analysis
        cross_validation_analysis(X_filtered, y_encoded, label_encoder)
        
        # Save improved model
        output_dir = Path(__file__).parent
        model_path = output_dir / "improved_lightgbm_classifier.pkl"
        encoder_path = output_dir / "improved_label_encoder.pkl"
        
        joblib.dump(model, model_path)
        joblib.dump(label_encoder, encoder_path)
        
        print(f"\n💾 Model artifacts saved:")
        print(f"   • Model: {model_path}")
        print(f"   • Encoder: {encoder_path}")
        
        print(f"\n🎉 Improved training pipeline completed!")
        print(f"📊 Performance summary:")
        print(f"   • Test accuracy: {results['accuracy']:.4f}")
        print(f"   • Classes: {len(label_encoder.classes_)}")
        print(f"   • Data retention: {filter_metadata['retention_rate']*100:.1f}%")
        
    except Exception as e:
        print(f"❌ Error in improved pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
