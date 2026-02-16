"""
Feature engineering module for customer data
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from ..config.settings import TARGET_COLUMN


class FeatureEngineer:
    """Class for feature engineering operations"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        
    def create_features(self, df):
        """
        Create engineered features from raw data
        
        Parameters:
        -----------
        df : pd.DataFrame
            Raw customer data
        
        Returns:
        --------
        pd.DataFrame
            Data with engineered features
        """
        df = df.copy()
        
        # Create interaction features
        df['balance_per_product'] = df['balance'] / (df['num_of_products'] + 1)
        df['tenure_age_ratio'] = df['tenure'] / (df['age'] + 1)
        df['salary_balance_ratio'] = df['estimated_salary'] / (df['balance'] + 1)
        
        # Create categorical features
        df['age_group'] = pd.cut(df['age'], 
                                 bins=[0, 30, 40, 50, 60, 100],
                                 labels=['<30', '30-40', '40-50', '50-60', '60+'])
        
        df['balance_category'] = pd.cut(df['balance'],
                                        bins=[0, 1000, 5000, 50000, np.inf],
                                        labels=['Low', 'Medium', 'High', 'Very High'])
        
        df['tenure_category'] = pd.cut(df['tenure'],
                                      bins=[0, 12, 24, 48, np.inf],
                                      labels=['New', 'Established', 'Long-term', 'Loyal'])
        
        # Create risk score
        df['risk_score'] = (
            (df['satisfaction_score'] < 3).astype(int) * 2 +
            (df['complaint_count'] > 2).astype(int) * 2 +
            (df['is_active_member'] == 0).astype(int) * 1 +
            (df['last_transaction_days'] > 30).astype(int) * 1
        )
        
        # Create engagement score
        df['engagement_score'] = (
            df['is_active_member'] * 2 +
            df['has_credit_card'] * 1 +
            df['num_of_products'] * 0.5 +
            (df['last_transaction_days'] < 7).astype(int) * 1.5
        )
        
        # Create value score
        df['value_score'] = (
            np.log1p(df['balance']) * 0.3 +
            np.log1p(df['estimated_salary']) * 0.2 +
            df['num_of_products'] * 0.3 +
            df['tenure'] * 0.2
        )
        
        return df
    
    def encode_features(self, df, fit=True):
        """
        Encode categorical features
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data with features
        fit : bool
            Whether to fit encoders (True for training, False for prediction)
        
        Returns:
        --------
        pd.DataFrame
            Data with encoded features
        """
        df = df.copy()
        
        # Encode gender
        if 'gender' in df.columns:
            if fit:
                self.label_encoders['gender'] = LabelEncoder()
                df['gender_encoded'] = self.label_encoders['gender'].fit_transform(df['gender'])
            else:
                df['gender_encoded'] = self.label_encoders['gender'].transform(df['gender'])
        
        # One-hot encode categorical features
        categorical_cols = ['age_group', 'balance_category', 'tenure_category']
        for col in categorical_cols:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col)
                df = pd.concat([df, dummies], axis=1)
                df = df.drop(columns=[col])
        
        return df
    
    def prepare_features(self, df, fit=True):
        """
        Prepare features for modeling
        
        Parameters:
        -----------
        df : pd.DataFrame
            Raw customer data
        fit : bool
            Whether to fit scalers (True for training, False for prediction)
        
        Returns:
        --------
        tuple
            (X, y) where X is features and y is target
        """
        # Create features
        df = self.create_features(df)
        
        # Encode features
        df = self.encode_features(df, fit=fit)
        
        # Select feature columns
        feature_cols = [col for col in df.columns 
                       if col not in [TARGET_COLUMN, 'customer_id', 'gender'] 
                       and not col.startswith('age_group') 
                       and not col.startswith('balance_category')
                       and not col.startswith('tenure_category')]
        
        # Add one-hot encoded columns
        for col in df.columns:
            if col.startswith('age_group_') or col.startswith('balance_category_') or col.startswith('tenure_category_'):
                if col not in feature_cols:
                    feature_cols.append(col)
        
        # Ensure gender_encoded is included
        if 'gender_encoded' in df.columns and 'gender_encoded' not in feature_cols:
            feature_cols.append('gender_encoded')
        
        X = df[feature_cols].copy()
        
        # Store feature names
        if fit:
            self.feature_names = feature_cols
        
        # Scale features
        if fit:
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns,
                index=X.index
            )
        
        # Get target if available
        y = df[TARGET_COLUMN] if TARGET_COLUMN in df.columns else None
        
        return X_scaled, y
    
    def save(self, filepath):
        """Save feature engineer to file"""
        import joblib
        joblib.dump({
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names
        }, filepath)
    
    def load(self, filepath):
        """Load feature engineer from file"""
        import joblib
        data = joblib.load(filepath)
        self.scaler = data['scaler']
        self.label_encoders = data['label_encoders']
        self.feature_names = data['feature_names']
