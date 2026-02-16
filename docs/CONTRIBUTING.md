# Contributing to Customer Analysis

Thank you for your interest in contributing to the Customer Analysis project!

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/yourusername/customer-analysis.git
   cd customer-analysis
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

## Code Style

- Follow PEP 8 style guide
- Use black for code formatting: `black src/ tests/`
- Maximum line length: 100 characters
- Run flake8 before committing: `flake8 src/ tests/`

## Testing

- Write tests for new features
- Run tests: `pytest tests/ -v`
- Ensure all tests pass before submitting PR

## Pull Request Process

1. Create a feature branch: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Add tests if applicable
4. Ensure all tests pass
5. Update documentation if needed
6. Submit a pull request

## Commit Messages

- Use clear, descriptive commit messages
- Reference issue numbers if applicable
