"""
Main training script for Predictive Customer Analysis Project
"""
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import pandas as pd
from sklearn.model_selection import train_test_split

from customer_analysis import (
    generate_customer_data,
    FeatureEngineer,
    CustomerChurnPredictor,
    CustomerSegmenter,
    create_powerbi_dataset,
    export_to_excel,
    create_visualizations
)
from customer_analysis.config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR,
    RANDOM_STATE, TEST_SIZE
)


def main():
    """Main execution function"""
    print("=" * 60)
    print("Predictive Customer Analysis Project")
    print("=" * 60)
    print()
    
    # Step 1: Generate or load data
    print("Step 1: Data Generation")
    print("-" * 60)
    data_file = RAW_DATA_DIR / "customer_data.csv"
    
    if data_file.exists():
        print(f"Loading existing data from {data_file}")
        df = pd.read_csv(data_file)
    else:
        print("Generating new customer data...")
        df = generate_customer_data(n_samples=10000)
        df.to_csv(data_file, index=False)
        print(f"Data saved to {data_file}")
    
    print(f"Data shape: {df.shape}")
    print(f"Churn rate: {df['churn'].mean()*100:.2f}%")
    print()
    
    # Step 2: Feature Engineering
    print("Step 2: Feature Engineering")
    print("-" * 60)
    feature_engineer = FeatureEngineer()
    X, y = feature_engineer.prepare_features(df, fit=True)
    
    print(f"Features shape: {X.shape}")
    print(f"Feature columns: {list(X.columns[:10])}...")
    print()
    
    # Save feature engineer
    feature_engineer_path = MODELS_DIR / "feature_engineer.pkl"
    feature_engineer.save(str(feature_engineer_path))
    
    # Step 3: Split data
    print("Step 3: Data Splitting")
    print("-" * 60)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print()
    
    # Step 4: Train Churn Prediction Model
    print("Step 4: Training Churn Prediction Model")
    print("-" * 60)
    churn_model = CustomerChurnPredictor()
    churn_model.train(X_train, y_train, use_smote=True)
    
    # Evaluate model
    print("\nModel Evaluation:")
    print("-" * 60)
    metrics = churn_model.evaluate(X_test, y_test)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1_score']:.4f}")
    print(f"ROC AUC:   {metrics['roc_auc']:.4f}")
    print(f"\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    print()
    
    # Save model
    churn_model_path = MODELS_DIR / "churn_model.pkl"
    churn_model.save(str(churn_model_path))
    
    # Step 5: Customer Segmentation
    print("Step 5: Customer Segmentation")
    print("-" * 60)
    segmenter = CustomerSegmenter(n_clusters=4)
    segmenter.train(X_train)
    
    # Predict segments for all data
    segments = segmenter.predict(X)
    df['segment'] = segments
    
    # Save segmentation model
    segmenter_path = MODELS_DIR / "segmenter_model.pkl"
    segmenter.save(str(segmenter_path))
    
    print(f"Segments distribution:")
    print(df['segment'].value_counts().sort_index())
    print()
    
    # Step 6: Generate Predictions
    print("Step 6: Generating Predictions")
    print("-" * 60)
    churn_proba = churn_model.predict_proba(X)
    churn_pred = churn_model.predict(X)
    
    predictions_df = pd.DataFrame({
        'customer_id': df['customer_id'].values,
        'churn_probability': churn_proba,
        'churn_prediction': churn_pred,
        'risk_level': pd.cut(
            churn_proba,
            bins=[0, 0.3, 0.6, 1.0],
            labels=['Low', 'Medium', 'High']
        )
    })
    
    print(f"High risk customers (prob > 0.6): {(churn_proba > 0.6).sum()}")
    print(f"Medium risk customers (0.3 < prob <= 0.6): {((churn_proba > 0.3) & (churn_proba <= 0.6)).sum()}")
    print(f"Low risk customers (prob <= 0.3): {(churn_proba <= 0.3).sum()}")
    print()
    
    # Step 7: Create Power BI Dataset
    print("Step 7: Creating Power BI Dataset")
    print("-" * 60)
    segment_df = pd.DataFrame({
        'customer_id': df['customer_id'].values,
        'segment': segments
    })
    
    powerbi_df = create_powerbi_dataset(
        df, predictions_df, churn_model.feature_importance, segment_df
    )
    
    # Save Power BI dataset
    powerbi_file = PROCESSED_DATA_DIR / "powerbi_dataset.csv"
    powerbi_df.to_csv(powerbi_file, index=False)
    print(f"Power BI dataset saved to {powerbi_file}")
    print()
    
    # Step 8: Export to Excel
    print("Step 8: Exporting to Excel")
    print("-" * 60)
    excel_file = RESULTS_DIR / "powerbi_export.xlsx"
    export_to_excel(
        powerbi_df,
        churn_model.feature_importance,
        metrics,
        str(excel_file)
    )
    print()
    
    # Step 9: Create Visualizations
    print("Step 9: Creating Visualizations")
    print("-" * 60)
    create_visualizations(df, predictions_df, churn_model.feature_importance)
    print()
    
    # Step 10: Summary Report
    print("=" * 60)
    print("SUMMARY REPORT")
    print("=" * 60)
    print(f"\nTotal Customers Analyzed: {len(df):,}")
    print(f"Churned Customers: {df['churn'].sum():,} ({df['churn'].mean()*100:.2f}%)")
    print(f"\nModel Performance:")
    print(f"  - Accuracy:  {metrics['accuracy']:.2%}")
    print(f"  - Precision: {metrics['precision']:.2%}")
    print(f"  - Recall:    {metrics['recall']:.2%}")
    print(f"  - F1 Score:  {metrics['f1_score']:.2%}")
    print(f"  - ROC AUC:   {metrics['roc_auc']:.2%}")
    print(f"\nRisk Analysis:")
    print(f"  - High Risk Customers:   {(churn_proba > 0.6).sum():,} ({(churn_proba > 0.6).mean()*100:.2f}%)")
    print(f"  - Medium Risk Customers: {((churn_proba > 0.3) & (churn_proba <= 0.6)).sum():,} ({((churn_proba > 0.3) & (churn_proba <= 0.6)).mean()*100:.2f}%)")
    print(f"  - Low Risk Customers:    {(churn_proba <= 0.3).sum():,} ({(churn_proba <= 0.3).mean()*100:.2f}%)")
    print(f"\nOutput Files:")
    print(f"  - Power BI CSV: {powerbi_file}")
    print(f"  - Power BI Excel: {excel_file}")
    print(f"  - Visualizations: {RESULTS_DIR}/")
    print(f"  - Models: {MODELS_DIR}/")
    print("\n" + "=" * 60)
    print("Project execution completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
