"""
Customer Analysis Package
Predictive customer analysis and churn prediction
"""

__version__ = "1.0.0"
__author__ = "Customer Analysis Team"

from .data.generator import generate_customer_data
from .features.engineering import FeatureEngineer
from .models.predictors import CustomerChurnPredictor, CustomerSegmenter
from .visualization.exporter import create_powerbi_dataset, export_to_excel, create_visualizations

__all__ = [
    'generate_customer_data',
    'FeatureEngineer',
    'CustomerChurnPredictor',
    'CustomerSegmenter',
    'create_powerbi_dataset',
    'export_to_excel',
    'create_visualizations',
]
