# Contributing to Simulacrum MCP

Thank you for your interest in contributing to Simulacrum MCP! This document provides guidelines and information for contributors.

## 🚀 Quick Start

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/simulacrum-mcp.git
   cd simulacrum-mcp
   ```

3. **Set up development environment**:
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install development dependencies
   pip install -e ".[dev]"

   # Run tests to ensure everything works
   pytest
   ```

4. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## 📋 Development Guidelines

### Code Quality Standards

Simulacrum MCP follows strict code quality standards:

- **SOLID Principles**: Single responsibility, Open/closed, Liskov substitution, Interface segregation, Dependency inversion
- **DRY (Don't Repeat Yourself)**: No code duplication
- **Clean Code**: Self-documenting code with meaningful names
- **Type Safety**: Full type hints and mypy compliance
- **Test Coverage**: 100% test coverage required

### Code Style

We use automated formatting tools:

```bash
# Format code
black simulacrum/
isort simulacrum/

# Type checking
mypy simulacrum/

# Linting
flake8 simulacrum/
```

### Commit Messages

Follow conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test additions/modifications
- `chore`: Maintenance tasks

Examples:
```
feat(dynamics): add adaptive time stepping
fix(validation): handle edge case in scenario validation
test(chaos): add integration tests for Lyapunov exponent
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=simulacrum --cov-report=html

# Run specific test file
pytest tests/test_dynamics.py

# Run tests matching pattern
pytest -k "test_simulation"

# Run tests in verbose mode
pytest -v
```

### Writing Tests

- Use `pytest` framework
- Place tests in `tests/` directory
- Name test files as `test_*.py`
- Use descriptive test names: `test_should_do_something_when_condition`
- Use fixtures from `tests/conftest.py`
- Aim for 100% code coverage

Example test structure:
```python
import pytest
from simulacrum.tools.dynamics import DynamicsSimulator

class TestDynamicsSimulator:
    def test_simulate_dynamics_with_valid_scenario(self, sample_scenario_data):
        simulator = DynamicsSimulator()
        result = simulator.simulate_dynamics(sample_scenario_data)

        assert result["status"] == "success"
        assert "variables" in result
```

### Property-Based Testing

We use `hypothesis` for property-based testing:

```python
from hypothesis import given, strategies as st

@given(
    initial_value=st.floats(min_value=0.1, max_value=10.0),
    growth_rate=st.floats(min_value=0.01, max_value=1.0)
)
def test_logistic_growth_properties(initial_value, growth_rate):
    # Test properties that should hold for any valid inputs
    pass
```

## 📚 Documentation

### Code Documentation

- Use docstrings for all public functions, classes, and methods
- Follow Google/NumPy docstring format
- Include type hints for function parameters and return values
- Document exceptions that may be raised

```python
def simulate_dynamics(self, scenario_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute system dynamics simulation.

    Args:
        scenario_data: Dictionary containing scenario definition with:
            - name: Scenario name (string)
            - variables: List of variable definitions (list)
            - equations: List of equation definitions (list)

    Returns:
        Dictionary containing simulation results and metadata

    Raises:
        ValidationError: If scenario data is invalid

    Example:
        >>> simulator.simulate_dynamics({...})
        {'status': 'success', 'variables': {...}}
    """
```

### README Updates

When adding new features:
1. Update the main README.md with new functionality
2. Add usage examples
3. Update installation instructions if needed
4. Add new badges or links if relevant

## 🔧 Architecture Guidelines

### Adding New Tools

When adding a new simulation/analysis tool:

1. **Create the tool class** in `simulacrum/tools/`
2. **Follow the interface pattern** - implement required methods
3. **Add comprehensive validation** using Pydantic models
4. **Write thorough tests** - unit tests + integration tests
5. **Update the MCP server** to expose the new tool
6. **Add documentation** and usage examples

### Interface Pattern

```python
from simulacrum.validation.errors import ValidationError

class NewTool:
    """Tool for specific analysis type."""

    def analyze_something(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform analysis.

        Args:
            input_data: Input data for analysis

        Returns:
            Analysis results

        Raises:
            ValidationError: If input is invalid
        """
        # Implementation
        pass
```

### Error Handling

- Use custom exceptions from `simulacrum.validation.errors`
- Provide clear, actionable error messages
- Include relevant context in error details
- Don't expose internal implementation details

## 🚦 Pull Request Process

1. **Update tests** for any changed functionality
2. **Ensure all tests pass**:
   ```bash
   pytest --cov=simulacrum --cov-report=term-missing
   ```

3. **Run code quality checks**:
   ```bash
   black simulacrum/ tests/
   isort simulacrum/ tests/
   mypy simulacrum/
   flake8 simulacrum/
   ```

4. **Update documentation** if needed

5. **Create a Pull Request** with:
   - Clear title describing the change
   - Detailed description of what was changed and why
   - Link to any relevant issues
   - Screenshots/demo if UI changes

6. **Address review feedback** promptly

## 🐛 Reporting Issues

When reporting bugs:
- Use the GitHub issue tracker
- Include detailed reproduction steps
- Provide sample input data that triggers the bug
- Include environment information (Python version, OS, etc.)
- Describe expected vs. actual behavior

## 📝 License

By contributing to Simulacrum MCP, you agree that your contributions will be licensed under the MIT License.

## 💬 Getting Help

- **Documentation**: Check the README and docstrings first
- **Issues**: Search existing issues before creating new ones
- **Discussions**: Use GitHub Discussions for questions and ideas

Thank you for contributing to Simulacrum MCP! 🎉
