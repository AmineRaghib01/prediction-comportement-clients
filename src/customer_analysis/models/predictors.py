"""
Predictive models for customer analysis
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from imblearn.over_sampling import SMOTE
import joblib
from ..config.settings import RANDOM_STATE, RANDOM_FOREST_PARAMS


class CustomerChurnPredictor:
    """Customer churn prediction model"""
    
    def __init__(self):
        self.model = RandomForestClassifier(**RANDOM_FOREST_PARAMS)
        self.feature_importance = None
        self.is_trained = False
        
    def train(self, X_train, y_train, use_smote=True):
        """
        Train the churn prediction model
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
        use_smote : bool
            Whether to use SMOTE for handling class imbalance
        """
        print("Training churn prediction model...")
        
        # Handle class imbalance with SMOTE
        if use_smote:
            print("Applying SMOTE to balance classes...")
            smote = SMOTE(random_state=RANDOM_STATE)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
            print(f"After SMOTE: {len(X_train_balanced)} samples (was {len(X_train)})")
        else:
            X_train_balanced, y_train_balanced = X_train, y_train
        
        # Train model
        self.model.fit(X_train_balanced, y_train_balanced)
        
        # Store feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.is_trained = True
        print("Model training completed!")
        
    def predict(self, X):
        """
        Predict churn probability
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features
        
        Returns:
        --------
        np.array
            Churn predictions (0 or 1)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Predict churn probability
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features
        
        Returns:
        --------
        np.array
            Churn probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict_proba(X)[:, 1]
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        
        Parameters:
        -----------
        X_test : pd.DataFrame
            Test features
        y_test : pd.Series
            Test target
        
        Returns:
        --------
        dict
            Evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        return metrics
    
    def save(self, filepath):
        """Save model to file"""
        joblib.dump({
            'model': self.model,
            'feature_importance': self.feature_importance,
            'is_trained': self.is_trained
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model from file"""
        data = joblib.load(filepath)
        self.model = data['model']
        self.feature_importance = data['feature_importance']
        self.is_trained = data['is_trained']
        print(f"Model loaded from {filepath}")


class CustomerSegmenter:
    """Customer segmentation using clustering"""
    
    def __init__(self, n_clusters=4):
        from sklearn.cluster import KMeans
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
        self.is_trained = False
        
    def train(self, X):
        """
        Train clustering model
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features for clustering
        """
        print(f"Training customer segmentation model with {self.n_clusters} clusters...")
        self.model.fit(X)
        self.is_trained = True
        print("Segmentation model training completed!")
    
    def predict(self, X):
        """
        Predict customer segments
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features
        
        Returns:
        --------
        np.array
            Cluster assignments
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict(X)
    
    def save(self, filepath):
        """Save model to file"""
        joblib.dump({
            'model': self.model,
            'n_clusters': self.n_clusters,
            'is_trained': self.is_trained
        }, filepath)
        print(f"Segmentation model saved to {filepath}")
    
    def load(self, filepath):
        """Load model from file"""
        data = joblib.load(filepath)
        self.model = data['model']
        self.n_clusters = data['n_clusters']
        self.is_trained = data['is_trained']
        print(f"Segmentation model loaded from {filepath}")
