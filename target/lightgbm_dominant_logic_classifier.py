#!/usr/bin/env python3
"""
Random Forest model training pipeline for dominant logic classification.
Uses UMAP-reduced vectors from Qdrant to predict dominant logic categories.
"""

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class DominantLogicClassifier:
    """Random Forest classifier for dominant logic prediction using UMAP vectors."""
    
    def __init__(self, collection_name: str = "target_test_umap10d"):
        """Initialize the classifier with Qdrant collection."""
        self.collection_name = collection_name
        self.client = QdrantClient(host="localhost", port=6333)
        self.label_encoder = LabelEncoder()
        self.model = None
        self.feature_importance = None
        
    def extract_training_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Step 1: Extract records with non-empty dominant logic from Qdrant.
        
        Returns:
            Tuple: (umap_vectors, dominant_logic_labels, passages)
        """
        print(f"📊 Step 1: Extracting training data from '{self.collection_name}'...")
        
        # Get all points with vectors and payloads
        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            limit=10000,
            with_payload=True,
            with_vectors=True
        )
        
        print(f"✅ Retrieved {len(points)} total points from Qdrant")
        
        # Filter for non-empty dominant logic
        umap_vectors = []
        dominant_logic_labels = []
        passages = []
        
        for point in points:
            dominant_logic = point.payload.get('dominant_logic', '').strip()
            
            # Only include records with non-empty dominant logic
            if dominant_logic and dominant_logic.lower() not in ['', 'none', 'null', 'unknown']:
                umap_vectors.append(point.vector)
                dominant_logic_labels.append(dominant_logic)
                passages.append(point.payload.get('passage', ''))
        
        umap_vectors = np.array(umap_vectors)
        
        print(f"📈 Filtered to {len(umap_vectors)} records with valid dominant logic")
        print(f"🎯 Vector shape: {umap_vectors.shape}")
        
        # Show distribution of dominant logic categories
        logic_counts = pd.Series(dominant_logic_labels).value_counts()
        print(f"\n📊 Dominant Logic Distribution:")
        for logic, count in logic_counts.head(10).items():
            print(f"   • {logic}: {count} samples")
        
        if len(logic_counts) > 10:
            print(f"   ... and {len(logic_counts) - 10} more categories")
        
        return umap_vectors, np.array(dominant_logic_labels), passages
    
    def prepare_train_test_split(self, X: np.ndarray, y: np.ndarray, 
                                test_size: float = 0.2, 
                                random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Step 2: Split data into train and test sets.
        
        Args:
            X: Feature vectors (UMAP embeddings)
            y: Target labels (dominant logic)
            test_size: Proportion of test set
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple: (X_train, X_test, y_train, y_test)
        """
        print(f"\n🔄 Step 2: Splitting data into train/test sets...")
        print(f"   Test size: {test_size * 100:.1f}%")
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y_encoded  # Maintain class distribution
        )
        
        print(f"✅ Data split completed:")
        print(f"   Train set: {X_train.shape[0]} samples")
        print(f"   Test set: {X_test.shape[0]} samples")
        print(f"   Features: {X_train.shape[1]} dimensions")
        print(f"   Classes: {len(self.label_encoder.classes_)} unique categories")
        
        return X_train, X_test, y_train, y_test
    
    def train_xgboost_model(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_test: np.ndarray, y_test: np.ndarray) -> None:
        """
        Step 3: Train XGBoost model on UMAP vectors.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features (for early stopping)
            y_test: Test labels (for early stopping)
        """
        print(f"\n🚀 Step 3: Training XGBoost model...")
        
        # XGBoost parameters
        params = {
            'objective': 'multi:softprob',
            'num_class': len(self.label_encoder.classes_),
            'eval_metric': 'mlogloss',
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'verbosity': 1
        }
        
        print(f"   Model parameters:")
        for key, value in params.items():
            print(f"     • {key}: {value}")
        
        # Create DMatrix objects
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvalid = xgb.DMatrix(X_test, label=y_test)
        
        # Train model with early stopping
        evallist = [(dtrain, 'train'), (dvalid, 'valid')]
        
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=evallist,
            early_stopping_rounds=50,
            verbose_eval=100
        )
        
        print(f"✅ Model training completed!")
        print(f"   Best iteration: {self.model.best_iteration}")
        print(f"   Best score: {self.model.best_score:.4f}")
        
        # Get feature importance
        self.feature_importance = self.model.get_score(importance_type='gain')
        # Convert to array format for plotting
        feature_names = [f'f{i}' for i in range(X_train.shape[1])]
        self.feature_importance = np.array([self.feature_importance.get(f'f{i}', 0) 
                                          for i in range(X_train.shape[1])])
        
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray, 
                      passages_test: List[str] = None) -> Dict[str, Any]:
        """
        Step 4: Evaluate model on test set and show classification metrics.
        
        Args:
            X_test: Test features
            y_test: Test labels
            passages_test: Test passages for error analysis
            
        Returns:
            Dict: Evaluation metrics and results
        """
        print(f"\n📊 Step 4: Evaluating model on test set...")
        
        # Make predictions
        dtest = xgb.DMatrix(X_test)
        y_pred_proba = self.model.predict(dtest, iteration_range=(0, self.model.best_iteration + 1))
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Convert back to original labels
        y_test_labels = self.label_encoder.inverse_transform(y_test)
        y_pred_labels = self.label_encoder.inverse_transform(y_pred)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        precision_macro = precision_score(y_test, y_pred, average='macro')
        recall_macro = recall_score(y_test, y_pred, average='macro')
        
        print(f"🎯 Classification Metrics:")
        print(f"   • Accuracy: {accuracy:.4f}")
        print(f"   • F1-Score (Macro): {f1_macro:.4f}")
        print(f"   • F1-Score (Weighted): {f1_weighted:.4f}")
        print(f"   • Precision (Macro): {precision_macro:.4f}")
        print(f"   • Recall (Macro): {recall_macro:.4f}")
        
        # Detailed classification report
        print(f"\n📋 Detailed Classification Report:")
        class_report = classification_report(
            y_test_labels, y_pred_labels, 
            output_dict=True, zero_division=0
        )
        
        # Print per-class metrics
        for class_name, metrics in class_report.items():
            if isinstance(metrics, dict) and class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                print(f"   {class_name}:")
                print(f"     Precision: {metrics['precision']:.3f}")
                print(f"     Recall: {metrics['recall']:.3f}")
                print(f"     F1-Score: {metrics['f1-score']:.3f}")
                print(f"     Support: {int(metrics['support'])}")
        
        # Confusion matrix
        self.plot_confusion_matrix(y_test_labels, y_pred_labels)
        
        # Feature importance plot
        self.plot_feature_importance()
        
        # Error analysis
        if passages_test:
            self.analyze_errors(y_test_labels, y_pred_labels, passages_test)
        
        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'classification_report': class_report,
            'y_true': y_test_labels,
            'y_pred': y_pred_labels,
            'y_pred_proba': y_pred_proba
        }
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Plot confusion matrix."""
        print(f"\n📈 Generating confusion matrix...")
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.label_encoder.classes_,
            yticklabels=self.label_encoder.classes_
        )
        plt.title('Confusion Matrix - Dominant Logic Classification')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('confusion_matrix_dominant_logic.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"💾 Confusion matrix saved as 'confusion_matrix_dominant_logic.png'")
    
    def plot_feature_importance(self) -> None:
        """Plot feature importance."""
        print(f"\n📊 Generating feature importance plot...")
        
        if self.feature_importance is None:
            print("⚠️  No feature importance available")
            return
        
        # Create feature importance DataFrame
        feature_names = [f'UMAP_dim_{i}' for i in range(len(self.feature_importance))]
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importance
        }).sort_values('importance', ascending=True)
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['feature'], importance_df['importance'])
        plt.title('Feature Importance - UMAP Dimensions')
        plt.xlabel('Importance (Gain)')
        plt.ylabel('UMAP Dimensions')
        plt.tight_layout()
        plt.savefig('feature_importance_umap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"💾 Feature importance plot saved as 'feature_importance_umap.png'")
    
    def analyze_errors(self, y_true: np.ndarray, y_pred: np.ndarray, 
                      passages: List[str]) -> None:
        """Analyze misclassified examples."""
        print(f"\n🔍 Error Analysis:")
        
        # Find misclassified examples
        errors = y_true != y_pred
        error_indices = np.where(errors)[0]
        
        print(f"   Total errors: {len(error_indices)}/{len(y_true)} ({len(error_indices)/len(y_true)*100:.1f}%)")
        
        if len(error_indices) > 0:
            print(f"\n❌ Sample misclassified examples:")
            for i, idx in enumerate(error_indices[:5]):  # Show first 5 errors
                print(f"   {i+1}. Passage: \"{passages[idx][:100]}...\"")
                print(f"      True: {y_true[idx]}")
                print(f"      Predicted: {y_pred[idx]}")
                print()

def main():
    """Main function to run the complete ML pipeline."""
    print("🎯 XGBoost Training Pipeline for Dominant Logic Classification")
    print("=" * 70)
    
    try:
        # Initialize classifier
        classifier = DominantLogicClassifier("target_test_umap10d")
        
        # Step 1: Extract training data
        X, y, passages = classifier.extract_training_data()
        
        if len(X) < 10:
            print("❌ Not enough training data (need at least 10 samples)")
            return
        
        # Step 2: Train/test split
        X_train, X_test, y_train, y_test = classifier.prepare_train_test_split(X, y)
        
        # Also split passages for error analysis
        passages_array = np.array(passages)
        _, passages_test, _, _ = train_test_split(
            passages_array, y, test_size=0.2, random_state=42, 
            stratify=classifier.label_encoder.transform(y)
        )
        
        # Step 3: Train XGBoost model
        classifier.train_xgboost_model(X_train, y_train, X_test, y_test)
        
        # Step 4: Evaluate model
        results = classifier.evaluate_model(X_test, y_test, passages_test.tolist())
        
        print(f"\n🎉 Training pipeline completed successfully!")
        print(f"📊 Final Results Summary:")
        print(f"   • Dataset size: {len(X)} samples")
        print(f"   • Classes: {len(classifier.label_encoder.classes_)} categories")
        print(f"   • Test accuracy: {results['accuracy']:.4f}")
        print(f"   • Test F1-score: {results['f1_weighted']:.4f}")
        print(f"   • Feature dimensions: {X.shape[1]}D UMAP vectors")
        
        print(f"\n💾 Generated files:")
        print(f"   • confusion_matrix_dominant_logic.png")
        print(f"   • feature_importance_umap.png")
        
    except Exception as e:
        print(f"❌ Error in ML pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
