#!/usr/bin/env python3
"""
LightGBM training pipeline for dominant logic classification using UMAP embeddings.
"""

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict
import time
from pathlib import Path
import joblib

def extract_training_data(collection_name: str = "target_test") -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Step 1: Extract records with non-empty dominant logic from Qdrant.
    
    Args:
        collection_name: Name of the Qdrant collection
        
    Returns:
        tuple: (umap_embeddings, dominant_logics, passages)
    """
    print(f"🔍 Step 1: Extracting training data from '{collection_name}'...")
    
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
    
    # Filter points with non-empty dominant logic and UMAP embeddings
    umap_embeddings = []
    dominant_logics = []
    passages = []
    
    for point in points:
        payload = point.payload
        dominant_logic = payload.get('dominant_logic', '').strip()
        umap_embedding = payload.get('umap_embedding', [])
        passage = payload.get('passage', '')
        
        # Only include if dominant logic is not empty and UMAP embedding exists
        if dominant_logic and len(umap_embedding) == 10:
            umap_embeddings.append(umap_embedding)
            dominant_logics.append(dominant_logic)
            passages.append(passage)
    
    umap_embeddings = np.array(umap_embeddings)
    dominant_logics = np.array(dominant_logics)
    
    print(f"✅ Filtered to {len(umap_embeddings)} records with non-empty dominant logic")
    print(f"📈 UMAP embeddings shape: {umap_embeddings.shape}")
    
    # Show class distribution
    unique_logics, counts = np.unique(dominant_logics, return_counts=True)
    print(f"🏷️  Class distribution:")
    for logic, count in zip(unique_logics, counts):
        print(f"   • {logic}: {count} samples")
    
    return umap_embeddings, dominant_logics, passages

def prepare_train_test_split(X: np.ndarray, y: np.ndarray, 
                           test_size: float = 0.2, 
                           random_state: int = 42,
                           min_samples_per_class: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, LabelEncoder]:
    """
    Step 2: Split data into train/test sets and encode labels.
    
    Args:
        X: Feature matrix (UMAP embeddings)
        y: Target labels (dominant logics)
        test_size: Proportion of test data
        random_state: Random seed
        min_samples_per_class: Minimum samples per class to include
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, label_encoder)
    """
    print(f"🔄 Step 2: Splitting data into train/test sets...")
    
    # Filter out classes with too few samples
    unique_classes, class_counts = np.unique(y, return_counts=True)
    valid_classes = unique_classes[class_counts >= min_samples_per_class]
    
    print(f"📊 Class filtering:")
    print(f"   • Total classes: {len(unique_classes)}")
    print(f"   • Classes with ≥{min_samples_per_class} samples: {len(valid_classes)}")
    
    if len(valid_classes) < len(unique_classes):
        # Filter data to only include classes with sufficient samples
        valid_mask = np.isin(y, valid_classes)
        X = X[valid_mask]
        y = y[valid_mask]
        
        excluded_classes = unique_classes[class_counts < min_samples_per_class]
        print(f"   ⚠️  Excluded classes: {excluded_classes.tolist()}")
        print(f"   ✅ Retained {len(X)} samples from {len(valid_classes)} classes")
    
    # Encode string labels to integers
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"📊 Final label encoding:")
    for i, class_name in enumerate(label_encoder.classes_):
        count = np.sum(y_encoded == i)
        print(f"   • {i}: {class_name} ({count} samples)")
    
    # Determine if we can use stratification
    min_class_count = np.min(np.bincount(y_encoded))
    use_stratify = min_class_count >= 2 and len(X) > 10
    
    # Split data
    if use_stratify:
        print(f"✅ Using stratified split to maintain class distribution")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y_encoded
        )
    else:
        print(f"⚠️  Using random split (stratification not possible)")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, 
            test_size=test_size, 
            random_state=random_state
        )
    
    print(f"✅ Data split complete:")
    print(f"   📚 Training set: {X_train.shape[0]} samples")
    print(f"   🧪 Test set: {X_test.shape[0]} samples")
    print(f"   🎯 Features: {X_train.shape[1]} dimensions (UMAP)")
    print(f"   🏷️  Classes: {len(label_encoder.classes_)}")
    
    return X_train, X_test, y_train, y_test, label_encoder

def train_lightgbm_model(X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray) -> lgb.LGBMClassifier:
    """
    Step 3: Train LightGBM model on UMAP vectors.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features (for early stopping)
        y_test: Test labels (for early stopping)
        
    Returns:
        lgb.LGBMClassifier: Trained model
    """
    print(f"🚀 Step 3: Training LightGBM model...")
    
    # Configure LightGBM parameters
    lgbm_params = {
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.1,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'random_state': 42
    }
    
    # Create and train model
    model = lgb.LGBMClassifier(
        **lgbm_params,
        n_estimators=100,
        early_stopping_rounds=10,
        verbose=-1
    )
    
    print(f"⏳ Training LightGBM model...")
    start_time = time.time()
    
    # Train with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
    )
    
    end_time = time.time()
    print(f"✅ Training completed in {end_time - start_time:.2f} seconds")
    print(f"🌳 Best iteration: {model.best_iteration_}")
    print(f"🎯 Best score: {model.best_score_['valid_0']['multi_logloss']:.4f}")
    
    return model

def evaluate_model(model: lgb.LGBMClassifier, 
                  X_test: np.ndarray, y_test: np.ndarray,
                  label_encoder: LabelEncoder,
                  save_plots: bool = True) -> Dict:
    """
    Step 4: Evaluate model and show classification metrics.
    
    Args:
        model: Trained LightGBM model
        X_test: Test features
        y_test: Test labels
        label_encoder: Label encoder for class names
        save_plots: Whether to save visualization plots
        
    Returns:
        dict: Evaluation metrics
    """
    print(f"📊 Step 4: Evaluating model on test set...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"🎯 Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Classification report
    class_names = label_encoder.classes_
    print(f"\n📈 Classification Report:")
    print("=" * 60)
    
    # Get unique classes present in test set
    unique_test_classes = np.unique(y_test)
    test_class_names = [class_names[i] for i in unique_test_classes]
    
    report = classification_report(y_test, y_pred, 
                                 target_names=test_class_names,
                                 labels=unique_test_classes,
                                 digits=4)
    print(report)
    
    # Confusion matrix
    print(f"\n🔍 Confusion Matrix:")
    print("=" * 40)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Feature importance
    print(f"\n🌟 Feature Importance (UMAP dimensions):")
    print("=" * 50)
    feature_importance = model.feature_importances_
    for i, importance in enumerate(feature_importance):
        print(f"   • UMAP_dim_{i}: {importance:.4f}")
    
    # Create visualizations
    if save_plots:
        create_evaluation_plots(y_test, y_pred, y_pred_proba, class_names, cm, feature_importance)
    
    # Return metrics summary
    return {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm,
        'feature_importance': feature_importance,
        'class_names': class_names
    }

def create_evaluation_plots(y_test: np.ndarray, y_pred: np.ndarray, 
                          y_pred_proba: np.ndarray, class_names: List[str],
                          cm: np.ndarray, feature_importance: np.ndarray) -> None:
    """
    Create and save evaluation plots.
    """
    print(f"📊 Creating evaluation plots...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('LightGBM Model Evaluation Results', fontsize=16, fontweight='bold')
    
    # 1. Confusion Matrix Heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0, 0])
    axes[0, 0].set_title('Confusion Matrix')
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Actual')
    
    # 2. Feature Importance
    feature_names = [f'UMAP_dim_{i}' for i in range(len(feature_importance))]
    axes[0, 1].barh(feature_names, feature_importance)
    axes[0, 1].set_title('Feature Importance (UMAP Dimensions)')
    axes[0, 1].set_xlabel('Importance')
    
    # 3. Class Distribution
    unique_test, counts_test = np.unique(y_test, return_counts=True)
    test_class_names = [class_names[i] for i in unique_test]
    axes[1, 0].bar(test_class_names, counts_test, alpha=0.7, label='Test Set')
    axes[1, 0].set_title('Test Set Class Distribution')
    axes[1, 0].set_xlabel('Class')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. Prediction Confidence Distribution
    max_proba = np.max(y_pred_proba, axis=1)
    axes[1, 1].hist(max_proba, bins=20, alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Prediction Confidence Distribution')
    axes[1, 1].set_xlabel('Max Probability')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].axvline(np.mean(max_proba), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(max_proba):.3f}')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('lightgbm_evaluation_results.png', dpi=300, bbox_inches='tight')
    print(f"💾 Plots saved as 'lightgbm_evaluation_results.png'")
    
    plt.show()

def save_model_and_encoder(model: lgb.LGBMClassifier, 
                          label_encoder: LabelEncoder,
                          model_path: str = "lightgbm_dominant_logic_classifier.pkl",
                          encoder_path: str = "label_encoder.pkl") -> None:
    """
    Save trained model and label encoder.
    """
    print(f"💾 Saving model and encoder...")
    
    # Save model
    joblib.dump(model, model_path)
    print(f"✅ Model saved as '{model_path}'")
    
    # Save label encoder
    joblib.dump(label_encoder, encoder_path)
    print(f"✅ Label encoder saved as '{encoder_path}'")

def main():
    """Main function to run the complete ML pipeline."""
    print("🎯 LightGBM Training Pipeline for Dominant Logic Classification")
    print("=" * 70)
    
    try:
        # Step 1: Extract training data
        X, y, passages = extract_training_data("target_test")
        
        if len(X) < 10:
            print("❌ Insufficient training data (need at least 10 samples)")
            return
        
        # Step 2: Prepare train/test split
        X_train, X_test, y_train, y_test, label_encoder = prepare_train_test_split(X, y)
        
        # Step 3: Train LightGBM model
        model = train_lightgbm_model(X_train, y_train, X_test, y_test)
        
        # Step 4: Evaluate model
        metrics = evaluate_model(model, X_test, y_test, label_encoder)
        
        # Save model and encoder
        save_model_and_encoder(model, label_encoder)
        
        print(f"\n🎉 Training pipeline completed successfully!")
        print(f"📊 Final Results:")
        print(f"   🎯 Test Accuracy: {metrics['accuracy']:.4f}")
        print(f"   📚 Training samples: {len(X_train)}")
        print(f"   🧪 Test samples: {len(X_test)}")
        print(f"   🏷️  Classes: {len(label_encoder.classes_)}")
        print(f"   🚀 Model: LightGBM with UMAP 10D embeddings")
        
        # Show sample predictions
        print(f"\n🔍 Sample Predictions:")
        print("=" * 50)
        for i in range(min(5, len(X_test))):
            pred_class = label_encoder.inverse_transform([y_test[i], model.predict([X_test[i]])[0]])
            print(f"   Sample {i+1}:")
            print(f"     True: {pred_class[0]}")
            print(f"     Pred: {pred_class[1]}")
            print(f"     Match: {'✅' if pred_class[0] == pred_class[1] else '❌'}")
        
    except Exception as e:
        print(f"❌ Error in training pipeline: {e}")
        raise

if __name__ == "__main__":
    main()
