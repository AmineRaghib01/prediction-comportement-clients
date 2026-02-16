"""
Visualization and Power BI export module
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ..config.settings import RESULTS_DIR

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def create_powerbi_dataset(df, predictions_df, feature_importance, segment_df=None):
    """
    Create comprehensive dataset for Power BI visualization
    
    Parameters:
    -----------
    df : pd.DataFrame
        Original customer data
    predictions_df : pd.DataFrame
        Predictions dataframe with churn probabilities
    feature_importance : pd.DataFrame
        Feature importance dataframe
    segment_df : pd.DataFrame, optional
        Customer segments dataframe
    
    Returns:
    --------
    pd.DataFrame
        Combined dataset for Power BI
    """
    # Merge original data with predictions
    powerbi_df = df.merge(
        predictions_df[['customer_id', 'churn_probability', 'churn_prediction', 'risk_level']],
        on='customer_id',
        how='left'
    )
    
    # Add segments if available
    if segment_df is not None:
        powerbi_df = powerbi_df.merge(
            segment_df[['customer_id', 'segment']],
            on='customer_id',
            how='left'
        )
    
    # Create risk categories
    powerbi_df['risk_category'] = pd.cut(
        powerbi_df['churn_probability'],
        bins=[0, 0.3, 0.6, 1.0],
        labels=['Low Risk', 'Medium Risk', 'High Risk']
    )
    
    # Create value segments
    powerbi_df['value_segment'] = pd.cut(
        powerbi_df['balance'],
        bins=[0, 1000, 10000, 50000, np.inf],
        labels=['Low Value', 'Medium Value', 'High Value', 'Premium']
    )
    
    # Create engagement segments
    powerbi_df['engagement_segment'] = pd.cut(
        powerbi_df['num_of_products'],
        bins=[0, 1, 2, 3, np.inf],
        labels=['Low Engagement', 'Medium Engagement', 'High Engagement', 'Very High Engagement']
    )
    
    return powerbi_df


def export_to_excel(powerbi_df, feature_importance, metrics_dict, filepath):
    """
    Export results to Excel for Power BI import
    
    Parameters:
    -----------
    powerbi_df : pd.DataFrame
        Power BI dataset
    feature_importance : pd.DataFrame
        Feature importance
    metrics_dict : dict
        Model evaluation metrics
    filepath : str
        Output file path
    """
    print(f"Exporting to Excel: {filepath}")
    
    with pd.ExcelWriter(filepath, engine='xlsxwriter') as writer:
        # Main dataset
        powerbi_df.to_excel(writer, sheet_name='Customer Data', index=False)
        
        # Feature importance
        feature_importance.to_excel(writer, sheet_name='Feature Importance', index=False)
        
        # Model metrics
        metrics_df = pd.DataFrame([metrics_dict])
        metrics_df.to_excel(writer, sheet_name='Model Metrics', index=False)
        
        # Summary statistics
        summary = powerbi_df.describe()
        summary.to_excel(writer, sheet_name='Summary Statistics')
        
        # Churn analysis
        churn_analysis = pd.DataFrame({
            'Metric': ['Total Customers', 'Churned Customers', 'Churn Rate', 
                      'High Risk Customers', 'High Risk Rate'],
            'Value': [
                len(powerbi_df),
                powerbi_df['churn'].sum(),
                powerbi_df['churn'].mean(),
                (powerbi_df['churn_probability'] > 0.6).sum(),
                (powerbi_df['churn_probability'] > 0.6).mean()
            ]
        })
        churn_analysis.to_excel(writer, sheet_name='Churn Analysis', index=False)
        
        # Segment analysis
        if 'segment' in powerbi_df.columns:
            segment_analysis = powerbi_df.groupby('segment').agg({
                'churn': 'mean',
                'balance': 'mean',
                'satisfaction_score': 'mean',
                'customer_id': 'count'
            }).rename(columns={'customer_id': 'count'})
            segment_analysis.to_excel(writer, sheet_name='Segment Analysis')
    
    print("Excel export completed!")


def create_visualizations(df, predictions_df, feature_importance, output_dir=None):
    """
    Create visualization plots
    
    Parameters:
    -----------
    df : pd.DataFrame
        Original customer data
    predictions_df : pd.DataFrame
        Predictions dataframe
    feature_importance : pd.DataFrame
        Feature importance
    output_dir : str, optional
        Output directory for plots (defaults to RESULTS_DIR)
    """
    if output_dir is None:
        output_dir = RESULTS_DIR
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Merge data
    viz_df = df.merge(
        predictions_df[['customer_id', 'churn_probability', 'churn_prediction']],
        on='customer_id',
        how='left'
    )
    
    # 1. Feature Importance Plot
    plt.figure(figsize=(10, 8))
    top_features = feature_importance.head(15)
    sns.barplot(data=top_features, y='feature', x='importance', palette='viridis')
    plt.title('Top 15 Most Important Features for Churn Prediction', fontsize=14, fontweight='bold')
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Churn Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Actual churn
    axes[0].pie(viz_df['churn'].value_counts(), 
                labels=['Retained', 'Churned'],
                autopct='%1.1f%%',
                colors=['#2ecc71', '#e74c3c'],
                startangle=90)
    axes[0].set_title('Actual Churn Distribution', fontsize=12, fontweight='bold')
    
    # Predicted churn probability distribution
    axes[1].hist(viz_df['churn_probability'], bins=30, color='#3498db', edgecolor='black')
    axes[1].axvline(viz_df['churn_probability'].mean(), color='red', linestyle='--', 
                    label=f'Mean: {viz_df["churn_probability"].mean():.2f}')
    axes[1].set_xlabel('Churn Probability', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].set_title('Churn Probability Distribution', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/churn_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Risk Analysis by Segment
    viz_df['risk_level'] = pd.cut(
        viz_df['churn_probability'],
        bins=[0, 0.3, 0.6, 1.0],
        labels=['Low', 'Medium', 'High']
    )
    
    plt.figure(figsize=(12, 6))
    risk_by_segment = pd.crosstab(viz_df['risk_level'], viz_df['churn'], normalize='index') * 100
    risk_by_segment.plot(kind='bar', stacked=True, color=['#2ecc71', '#e74c3c'])
    plt.title('Churn Rate by Risk Level', fontsize=14, fontweight='bold')
    plt.xlabel('Risk Level', fontsize=12)
    plt.ylabel('Percentage', fontsize=12)
    plt.legend(['Retained', 'Churned'], loc='upper right')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/risk_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Customer Characteristics Analysis
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Balance vs Churn
    sns.boxplot(data=viz_df, x='churn', y='balance', ax=axes[0, 0])
    axes[0, 0].set_title('Balance Distribution by Churn Status', fontweight='bold')
    axes[0, 0].set_xlabel('Churn')
    axes[0, 0].set_ylabel('Balance')
    
    # Satisfaction Score vs Churn
    sns.boxplot(data=viz_df, x='churn', y='satisfaction_score', ax=axes[0, 1])
    axes[0, 1].set_title('Satisfaction Score by Churn Status', fontweight='bold')
    axes[0, 1].set_xlabel('Churn')
    axes[0, 1].set_ylabel('Satisfaction Score')
    
    # Tenure vs Churn
    sns.boxplot(data=viz_df, x='churn', y='tenure', ax=axes[1, 0])
    axes[1, 0].set_title('Tenure Distribution by Churn Status', fontweight='bold')
    axes[1, 0].set_xlabel('Churn')
    axes[1, 0].set_ylabel('Tenure (months)')
    
    # Products vs Churn
    product_churn = pd.crosstab(viz_df['num_of_products'], viz_df['churn'], normalize='index') * 100
    product_churn.plot(kind='bar', ax=axes[1, 1], color=['#2ecc71', '#e74c3c'])
    axes[1, 1].set_title('Churn Rate by Number of Products', fontweight='bold')
    axes[1, 1].set_xlabel('Number of Products')
    axes[1, 1].set_ylabel('Percentage')
    axes[1, 1].legend(['Retained', 'Churned'])
    axes[1, 1].tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/customer_characteristics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to {output_dir}/")
