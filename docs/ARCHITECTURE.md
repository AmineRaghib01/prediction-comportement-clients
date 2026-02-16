# Architecture Documentation

## Overview

The Customer Analysis project follows a modular, professional Python package structure designed for maintainability, testability, and scalability.

## Package Structure

### `src/customer_analysis/`

Main package containing all core functionality:

- **`config/`**: Configuration management
  - `settings.py`: All project settings, paths, and hyperparameters
  
- **`data/`**: Data generation and loading
  - `generator.py`: Synthetic customer data generation
  
- **`features/`**: Feature engineering
  - `engineering.py`: Feature creation and transformation
  
- **`models/`**: Predictive models
  - `predictors.py`: Churn prediction and customer segmentation models
  
- **`visualization/`**: Visualization and exports
  - `exporter.py`: Power BI exports and plot generation

## Data Flow

```
Raw Data → Feature Engineering → Model Training → Predictions → Visualization → Power BI Export
```

## Key Design Decisions

1. **Modular Structure**: Each component is in its own module for easy testing and maintenance
2. **Configuration Management**: Centralized configuration in `config/settings.py`
3. **Path Management**: Uses `pathlib.Path` for cross-platform compatibility
4. **Model Persistence**: Models saved using joblib for easy loading/reloading
5. **Type Hints**: Code includes type hints for better IDE support and documentation

## Dependencies

- **Core**: pandas, numpy, scikit-learn
- **Visualization**: matplotlib, seaborn
- **Imbalance Handling**: imbalanced-learn
- **Export**: openpyxl, xlsxwriter
- **Testing**: pytest, pytest-cov

## Extension Points

- Add new models in `models/predictors.py`
- Add new features in `features/engineering.py`
- Add new visualizations in `visualization/exporter.py`
- Modify configuration in `config/settings.py`
