"""
Tests for feature engineering module
"""
import pytest
import pandas as pd
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from customer_analysis.features.engineering import FeatureEngineer
from customer_analysis.data.generator import generate_customer_data


class TestFeatureEngineer:
    """Test cases for feature engineering"""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing"""
        return generate_customer_data(n_samples=100)
    
    @pytest.fixture
    def feature_engineer(self):
        """Create feature engineer instance"""
        return FeatureEngineer()
    
    def test_create_features(self, feature_engineer, sample_data):
        """Test feature creation"""
        df = feature_engineer.create_features(sample_data)
        assert 'risk_score' in df.columns
        assert 'engagement_score' in df.columns
        assert 'value_score' in df.columns
    
    def test_prepare_features_shape(self, feature_engineer, sample_data):
        """Test that prepared features have correct shape"""
        X, y = feature_engineer.prepare_features(sample_data, fit=True)
        assert X.shape[0] == len(sample_data)
        assert X.shape[1] > 0
        assert y.shape[0] == len(sample_data)
    
    def test_prepare_features_no_target(self, feature_engineer, sample_data):
        """Test feature preparation without target"""
        sample_data_no_target = sample_data.drop(columns=['churn'])
        X, y = feature_engineer.prepare_features(sample_data_no_target, fit=True)
        assert y is None
