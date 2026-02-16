# Customer Analysis - Predictive Customer Churn Analysis

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A professional Python package for predictive customer analysis, churn prediction, and customer segmentation. This project uses scikit-learn for machine learning models and provides Power BI integration for data visualization.

## Features

- **Synthetic Data Generation**: Generate realistic customer data for analysis
- **Advanced Feature Engineering**: Create sophisticated features from raw data
- **Predictive Modeling**:
  - Customer churn prediction using Random Forest
  - Customer segmentation using K-Means clustering
- **Power BI Integration**: Export data and visualizations for Power BI dashboards
- **Professional Structure**: Well-organized, tested, and documented codebase

## Project Structure

```
customer-analysis/
├── src/
│   └── customer_analysis/          # Main package
│       ├── __init__.py
│       ├── config/                 # Configuration module
│       │   ├── __init__.py
│       │   └── settings.py
│       ├── data/                   # Data generation
│       │   ├── __init__.py
│       │   └── generator.py
│       ├── features/               # Feature engineering
│       │   ├── __init__.py
│       │   └── engineering.py
│       ├── models/                 # Predictive models
│       │   ├── __init__.py
│       │   └── predictors.py
│       └── visualization/          # Visualization & exports
│           ├── __init__.py
│           └── exporter.py
├── scripts/                        # Executable scripts
│   └── train.py                    # Main training script
├── tests/                          # Unit tests
│   ├── __init__.py
│   ├── test_data.py
│   └── test_features.py
├── docs/                           # Documentation
│   └── CONTRIBUTING.md
├── data/                           # Data directories (auto-created)
│   ├── raw/
│   └── processed/
├── models/                         # Saved models (auto-created)
├── results/                        # Results & visualizations (auto-created)
├── setup.py                        # Package setup
├── pyproject.toml                  # Modern Python project config
├── requirements.txt                # Production dependencies
├── requirements-dev.txt            # Development dependencies
├── Makefile                        # Common commands
├── LICENSE                         # MIT License
└── README.md                       # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Install Package

```bash
# Clone the repository
git clone https://github.com/yourusername/customer-analysis.git
cd customer-analysis

# Install package
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Install Dependencies Only

```bash
pip install -r requirements.txt
```

## Quick Start

### Run Complete Pipeline

```bash
# Using the script
python scripts/train.py

# Or using Makefile
make run
```

This will:
1. Generate or load customer data
2. Perform feature engineering
3. Train churn prediction model
4. Perform customer segmentation
5. Generate predictions
6. Create visualizations
7. Export data for Power BI

### Use as Python Package

```python
from customer_analysis import (
    generate_customer_data,
    FeatureEngineer,
    CustomerChurnPredictor,
    CustomerSegmenter
)

# Generate data
df = generate_customer_data(n_samples=10000)

# Feature engineering
fe = FeatureEngineer()
X, y = fe.prepare_features(df, fit=True)

# Train model
model = CustomerChurnPredictor()
model.train(X_train, y_train)

# Make predictions
predictions = model.predict_proba(X_test)
```

## Output Files

After execution, you'll find:

- **`data/processed/powerbi_dataset.csv`**: Complete dataset for Power BI
- **`results/powerbi_export.xlsx`**: Excel export with multiple sheets:
  - Customer Data: Customer data with predictions
  - Feature Importance: Feature importance scores
  - Model Metrics: Model evaluation metrics
  - Summary Statistics: Descriptive statistics
  - Churn Analysis: Churn analysis summary
  - Segment Analysis: Segment analysis
- **`results/*.png`**: Visualization plots
- **`models/*.pkl`**: Saved models for reuse

## Models

### Churn Prediction

- **Algorithm**: Random Forest Classifier
- **Features**: Engineered features from customer data
- **Class Imbalance**: Handled with SMOTE
- **Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC

### Customer Segmentation

- **Algorithm**: K-Means Clustering
- **Clusters**: 4 customer segments
- **Purpose**: Identify similar customer behaviors

## Feature Engineering

The feature engineering module creates:

- **Interaction Features**: `balance_per_product`, `tenure_age_ratio`, `salary_balance_ratio`
- **Categorical Features**: `age_group`, `balance_category`, `tenure_category`
- **Composite Scores**:
  - `risk_score`: Churn risk indicator
  - `engagement_score`: Customer engagement level
  - `value_score`: Customer value metric

## Development

### Setup Development Environment

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Or manually
pip install -r requirements-dev.txt
```

### Run Tests

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=src/customer_analysis --cov-report=html

# Using Makefile
make test
```

### Code Quality

```bash
# Format code
black src/ tests/ scripts/
make format

# Lint code
flake8 src/ tests/ scripts/
make lint
```

### Common Commands

```bash
make help          # Show all available commands
make install       # Install package
make install-dev   # Install with dev dependencies
make test          # Run tests
make lint          # Run linters
make format        # Format code
make clean         # Clean build artifacts
make run           # Run training script
```

## Configuration

Modify `src/customer_analysis/config/settings.py` to adjust:

- Model hyperparameters
- File paths
- Feature engineering parameters
- Dataset sizes

## Power BI Integration

The exported Excel file (`results/powerbi_export.xlsx`) contains multiple sheets ready for Power BI:

1. Import the Excel file into Power BI
2. Create relationships between tables
3. Build dashboards with:
   - Churn prediction metrics
   - Customer segmentation analysis
   - Risk level distributions
   - Feature importance charts

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Customer Analysis Team

## Acknowledgments

- scikit-learn for machine learning algorithms
- pandas and numpy for data manipulation
- matplotlib and seaborn for visualizations

## Support

For questions or issues, please open an issue on the GitHub repository.
