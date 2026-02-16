"""
Tests for data generation module
"""
import pytest
import pandas as pd
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from customer_analysis.data.generator import generate_customer_data


class TestDataGenerator:
    """Test cases for data generation"""
    
    def test_generate_customer_data_shape(self):
        """Test that generated data has correct shape"""
        df = generate_customer_data(n_samples=1000)
        assert df.shape[0] == 1000
        assert df.shape[1] > 10  # Should have multiple columns
    
    def test_generate_customer_data_columns(self):
        """Test that generated data has required columns"""
        df = generate_customer_data(n_samples=100)
        required_columns = [
            'customer_id', 'age', 'gender', 'tenure', 'balance',
            'num_of_products', 'has_credit_card', 'is_active_member',
            'estimated_salary', 'satisfaction_score', 'complaint_count',
            'last_transaction_days', 'churn'
        ]
        for col in required_columns:
            assert col in df.columns
    
    def test_churn_values(self):
        """Test that churn values are binary"""
        df = generate_customer_data(n_samples=100)
        assert df['churn'].isin([0, 1]).all()
    
    def test_data_types(self):
        """Test that data types are correct"""
        df = generate_customer_data(n_samples=100)
        assert df['customer_id'].dtype in ['int64', 'int32']
        assert df['age'].dtype in ['int64', 'int32']
        assert df['churn'].dtype in ['int64', 'int32']
