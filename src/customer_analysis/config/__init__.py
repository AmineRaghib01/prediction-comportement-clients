"""
Configuration module for Customer Analysis
"""
from .settings import (
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    VALIDATION_SIZE,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    RANDOM_FOREST_PARAMS,
    XGBOOST_PARAMS,
)

__all__ = [
    'DATA_DIR',
    'RAW_DATA_DIR',
    'PROCESSED_DATA_DIR',
    'MODELS_DIR',
    'RESULTS_DIR',
    'RANDOM_STATE',
    'TEST_SIZE',
    'VALIDATION_SIZE',
    'FEATURE_COLUMNS',
    'TARGET_COLUMN',
    'RANDOM_FOREST_PARAMS',
    'XGBOOST_PARAMS',
]
