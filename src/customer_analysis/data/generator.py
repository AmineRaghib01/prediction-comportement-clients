"""
Generate synthetic customer data for predictive analysis
"""
import pandas as pd
import numpy as np
from ..config.settings import RAW_DATA_DIR, RANDOM_STATE

np.random.seed(RANDOM_STATE)


def generate_customer_data(n_samples=10000):
    """
    Generate synthetic customer data with realistic patterns
    
    Parameters:
    -----------
    n_samples : int
        Number of customer records to generate
    
    Returns:
    --------
    pd.DataFrame
        Generated customer data
    """
    print(f"Generating {n_samples} customer records...")
    
    # Basic demographics
    age = np.random.normal(40, 15, n_samples)
    age = np.clip(age, 18, 80).astype(int)
    
    gender = np.random.choice(['Male', 'Female'], n_samples, p=[0.52, 0.48])
    
    # Customer tenure (in months)
    tenure = np.random.exponential(24, n_samples).astype(int)
    tenure = np.clip(tenure, 0, 120)
    
    # Account balance
    balance = np.random.lognormal(8, 1.5, n_samples)
    balance = np.clip(balance, 0, 250000)
    
    # Number of products
    num_of_products = np.random.choice([1, 2, 3, 4], n_samples, p=[0.4, 0.35, 0.2, 0.05])
    
    # Credit card ownership
    has_credit_card = np.random.choice([0, 1], n_samples, p=[0.45, 0.55])
    
    # Active membership
    is_active_member = np.random.choice([0, 1], n_samples, p=[0.35, 0.65])
    
    # Estimated salary
    estimated_salary = np.random.normal(50000, 20000, n_samples)
    estimated_salary = np.clip(estimated_salary, 10000, 200000)
    
    # Satisfaction score (1-5)
    satisfaction_score = np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.15, 0.2, 0.3, 0.25])
    
    # Complaint count
    complaint_count = np.random.poisson(0.5, n_samples)
    complaint_count = np.clip(complaint_count, 0, 5)
    
    # Days since last transaction
    last_transaction_days = np.random.exponential(7, n_samples).astype(int)
    last_transaction_days = np.clip(last_transaction_days, 0, 90)
    
    # Create DataFrame
    df = pd.DataFrame({
        'customer_id': range(1, n_samples + 1),
        'age': age,
        'gender': gender,
        'tenure': tenure,
        'balance': balance,
        'num_of_products': num_of_products,
        'has_credit_card': has_credit_card,
        'is_active_member': is_active_member,
        'estimated_salary': estimated_salary,
        'satisfaction_score': satisfaction_score,
        'complaint_count': complaint_count,
        'last_transaction_days': last_transaction_days
    })
    
    # Generate churn target with realistic patterns
    # Higher churn probability for:
    # - Low satisfaction scores
    # - High complaint count
    # - Inactive members
    # - Long time since last transaction
    # - Low balance
    churn_probability = (
        0.1 +  # Base probability
        (6 - df['satisfaction_score']) * 0.05 +  # Lower satisfaction = higher churn
        df['complaint_count'] * 0.08 +  # More complaints = higher churn
        (1 - df['is_active_member']) * 0.15 +  # Inactive = higher churn
        (df['last_transaction_days'] > 30).astype(int) * 0.12 +  # No recent activity
        (df['balance'] < 1000).astype(int) * 0.08 -  # Low balance
        (df['tenure'] > 24).astype(int) * 0.05  # Longer tenure = lower churn
    )
    
    churn_probability = np.clip(churn_probability, 0, 1)
    df['churn'] = np.random.binomial(1, churn_probability, n_samples)
    
    # Add some noise to make it more realistic
    df['balance'] = df['balance'] + np.random.normal(0, df['balance'] * 0.05, n_samples)
    df['balance'] = np.clip(df['balance'], 0, 250000)
    
    print(f"Generated data with {df['churn'].sum()} churned customers ({df['churn'].mean()*100:.2f}%)")
    
    return df
