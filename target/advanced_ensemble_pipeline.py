#!/usr/bin/env python3
"""
Advanced ensemble approach for dominant logic classification.
Combines multiple models and uses advanced preprocessing techniques.
"""

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import lightgbm as lgb
import xgboost as xgb
from typing import Tuple, List, Dict
import time
from pathlib import Path
import joblib
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

def advanced_data_preprocessing(collection_name: str = "target_test") -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Advanced data preprocessing with feature engineering.
    """
    print(f"🔍 Advanced Data Preprocessing from '{collection_name}'...")
    
    # Connect to Qdrant
    client = QdrantClient(host="localhost", port=6333)
    points, _ = client.scroll(
        collection_name=collection_name,
        limit=10000,
        with_payload=True,
        with_vectors=False
    )
    
    print(f"📊 Retrieved {len(points)} total points from Qdrant")
    
    # Extract data with better label cleaning
    umap_embeddings = []
    dominant_logics = []
    passages = []
    
    for point in points:
        umap_vector = point.payload.get('umap_embedding')
        dominant_logic = point.payload.get('dominant_logic', '').strip()
        
        if umap_vector and dominant_logic:
            # Advanced label cleaning and grouping
            cleaned_logic = clean_and_group_logic(dominant_logic)
            if cleaned_logic:
                umap_embeddings.append(umap_vector)
                dominant_logics.append(cleaned_logic)
                passages.append(point.payload.get('passage', ''))
    
    X = np.array(umap_embeddings)
    y = np.array(dominant_logics)
    
    # Feature engineering: add derived features
    X_enhanced = feature_engineering(X)
    
    print(f"✅ Preprocessed {len(X_enhanced)} samples")
    print(f"📈 Enhanced features: {X_enhanced.shape[1]}D (original: {X.shape[1]}D)")
    
    return X_enhanced, y, passages

def clean_and_group_logic(logic: str) -> str:
    """
    Advanced logic cleaning and grouping strategy.
    """
    logic = logic.strip().upper()
    
    # Skip unclear labels
    skip_terms = ['NOT YET IDENTIFIED', 'UNK', 'UNKNOWN', 'DOMINANT LOGICS', 'LOCAL', 'SCIENTIFIC METHOD']
    if any(term in logic for term in skip_terms) or len(logic) < 3:
        return None
    
    # Main categories with better grouping
    if 'FINANCIAL' in logic and 'PERFORMANCE' in logic:
        return 'FINANCIAL_PERFORMANCE'
    elif 'CERTAINTY' in logic and 'FINANCIAL' not in logic:
        return 'CERTAINTY'  
    elif 'ENTREPRENEUR' in logic:
        return 'ENTREPRENEUR'
    elif 'EXPERIMENT' in logic or 'ITERATE' in logic:
        return 'EXPERIMENTATION'
    elif 'RULES' in logic:
        return 'RULES_BASED'
    elif 'KNOWLEDGE' in logic or 'FIRST-HAND' in logic:
        return 'KNOWLEDGE_BASED'
    elif 'UNIQUE' in logic:
        return 'UNIQUE_APPROACH'
    else:
        return 'OTHER'

def feature_engineering(X: np.ndarray) -> np.ndarray:
    """
    Create additional features from UMAP embeddings.
    """
    print(f"🛠️  Feature engineering...")
    
    # Original features
    features = [X]
    
    # Statistical features
    # 1. Magnitude of embedding vector
    magnitude = np.linalg.norm(X, axis=1).reshape(-1, 1)
    features.append(magnitude)
    
    # 2. Mean and std of each sample
    mean_vals = np.mean(X, axis=1).reshape(-1, 1)
    std_vals = np.std(X, axis=1).reshape(-1, 1)
    features.extend([mean_vals, std_vals])
    
    # 3. Min and max values
    min_vals = np.min(X, axis=1).reshape(-1, 1)
    max_vals = np.max(X, axis=1).reshape(-1, 1)
    features.extend([min_vals, max_vals])
    
    # 4. Pairwise products of top dimensions (interaction features)
    # Use top 3 most important dimensions based on previous analysis
    top_dims = [1, 2, 5]  # From previous feature importance
    for i, dim1 in enumerate(top_dims):
        for dim2 in top_dims[i+1:]:
            interaction = (X[:, dim1] * X[:, dim2]).reshape(-1, 1)
            features.append(interaction)
    
    # Combine all features
    X_enhanced = np.hstack(features)
    
    print(f"   Added {X_enhanced.shape[1] - X.shape[1]} derived features")
    
    return X_enhanced

def create_ensemble_model(n_classes: int) -> VotingClassifier:
    """
    Create an ensemble of different models.
    """
    print(f"🤖 Creating ensemble model...")
    
    # Individual models with different strengths
    models = [
        ('lgb', lgb.LGBMClassifier(
            objective='multiclass',
            num_class=n_classes,
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            class_weight='balanced',
            random_state=42,
            verbose=-1
        )),
        ('rf', RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )),
        ('xgb', xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=n_classes,
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            random_state=42,
            eval_metric='mlogloss',
            verbosity=0
        )),
        ('lr', LogisticRegression(
            class_weight='balanced',
            random_state=42,
            max_iter=1000,
            multi_class='multinomial'
        ))
    ]
    
    # Create voting ensemble
    ensemble = VotingClassifier(
        estimators=models,
        voting='soft',  # Use probability voting
        n_jobs=-1
    )
    
    print(f"   Ensemble models: {[name for name, _ in models]}")
    
    return ensemble

def train_and_evaluate_advanced(X_train: np.ndarray, y_train: np.ndarray,
                               X_test: np.ndarray, y_test: np.ndarray,
                               label_encoder: LabelEncoder) -> Dict:
    """
    Train advanced ensemble and evaluate.
    """
    print(f"\n🚀 Training Advanced Ensemble Model...")
    
    # Scale features for some models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create ensemble
    ensemble = create_ensemble_model(len(label_encoder.classes_))
    
    # Train ensemble
    print(f"⏳ Training ensemble (this may take a moment)...")
    start_time = time.time()
    ensemble.fit(X_train_scaled, y_train)
    end_time = time.time()
    
    print(f"✅ Ensemble training completed in {end_time - start_time:.2f} seconds")
    
    # Predictions
    y_pred = ensemble.predict(X_test_scaled)
    y_pred_proba = ensemble.predict_proba(X_test_scaled)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    print(f"🎯 Ensemble Results:")
    print(f"   • Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   • F1-Score (Macro): {f1_macro:.4f}")
    print(f"   • F1-Score (Weighted): {f1_weighted:.4f}")
    
    # Detailed report
    y_test_labels = label_encoder.inverse_transform(y_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred)
    
    print(f"\n📈 Detailed Classification Report:")
    print("=" * 60)
    report = classification_report(y_test_labels, y_pred_labels, zero_division=0)
    print(report)
    
    # Individual model performance
    print(f"\n🔍 Individual Model Performance:")
    for name, model in ensemble.named_estimators_.items():
        if hasattr(model, 'predict'):
            individual_pred = model.predict(X_test_scaled)
            individual_acc = accuracy_score(y_test, individual_pred)
            print(f"   • {name.upper()}: {individual_acc:.4f}")
    
    return {
        'ensemble': ensemble,
        'scaler': scaler,
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'y_pred': y_pred_labels,
        'y_true': y_test_labels,
        'y_pred_proba': y_pred_proba
    }

def cross_validation_ensemble(X: np.ndarray, y: np.ndarray, n_classes: int) -> None:
    """
    Cross-validation for ensemble model.
    """
    print(f"\n📊 Ensemble Cross-Validation...")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create ensemble
    ensemble = create_ensemble_model(n_classes)
    
    # Perform CV
    cv_scores = []
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled, y)):
        X_train_cv, X_val_cv = X_scaled[train_idx], X_scaled[val_idx]
        y_train_cv, y_val_cv = y[train_idx], y[val_idx]
        
        ensemble_cv = create_ensemble_model(n_classes)
        ensemble_cv.fit(X_train_cv, y_train_cv)
        score = ensemble_cv.score(X_val_cv, y_val_cv)
        cv_scores.append(score)
        print(f"   Fold {fold+1}: {score:.4f}")
    
    print(f"🎯 CV Results: {np.mean(cv_scores):.4f} (±{np.std(cv_scores)*2:.4f})")

def main():
    """
    Main function for advanced ensemble approach.
    """
    print("🎯 Advanced Ensemble Pipeline for Dominant Logic Classification")
    print("=" * 80)
    
    try:
        # Step 1: Advanced preprocessing
        X, y, passages = advanced_data_preprocessing()
        
        if len(X) < 20:
            print("❌ Insufficient data")
            return
        
        # Filter classes
        class_counts = Counter(y)
        valid_classes = [cls for cls, count in class_counts.items() if count >= 3]
        mask = np.array([label in valid_classes for label in y])
        
        X_filtered = X[mask]
        y_filtered = y[mask]
        passages_filtered = [passages[i] for i in range(len(passages)) if mask[i]]
        
        print(f"\n📊 Final class distribution:")
        final_counts = Counter(y_filtered)
        for cls, count in final_counts.most_common():
            print(f"   • {cls}: {count} samples")
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y_filtered)
        
        print(f"\nDataset: {len(X_filtered)} samples, {X_filtered.shape[1]} features, {len(label_encoder.classes_)} classes")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_filtered, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded
        )
        
        print(f"Split: {len(X_train)} train, {len(X_test)} test")
        
        # Train and evaluate ensemble
        results = train_and_evaluate_advanced(X_train, y_train, X_test, y_test, label_encoder)
        
        # Cross-validation
        cross_validation_ensemble(X_filtered, y_encoded, len(label_encoder.classes_))
        
        # Save models
        output_dir = Path(__file__).parent
        ensemble_path = output_dir / "advanced_ensemble_classifier.pkl"
        scaler_path = output_dir / "feature_scaler.pkl"
        encoder_path = output_dir / "advanced_label_encoder.pkl"
        
        joblib.dump(results['ensemble'], ensemble_path)
        joblib.dump(results['scaler'], scaler_path)
        joblib.dump(label_encoder, encoder_path)
        
        print(f"\n💾 Advanced models saved:")
        print(f"   • Ensemble: {ensemble_path}")
        print(f"   • Scaler: {scaler_path}")
        print(f"   • Encoder: {encoder_path}")
        
        print(f"\n🎉 Advanced ensemble pipeline completed!")
        print(f"📊 Best Results:")
        print(f"   • Accuracy: {results['accuracy']:.4f}")
        print(f"   • F1-Weighted: {results['f1_weighted']:.4f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
