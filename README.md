# Simulacrum MCP

## Reality Engine for AI Systems

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**Simulacrum** is a comprehensive toolkit that transforms AI systems into sophisticated reality simulators. By providing advanced mathematical modeling, system dynamics simulation, and chaos analysis, it enables AI to understand and predict complex real-world phenomena through rigorous computational methods.

## 🌟 Key Features

### Core Simulation Tools
- **System Dynamics Simulation** - Model complex systems with differential equations and feedback loops
- **Scenario Comparison** - A/B testing for reality with sophisticated similarity metrics
- **Feedback Loop Analysis** - Identify reinforcing/balancing loops and their systemic impact
- **Belief Dynamics** - Theory of Mind simulation for social and cognitive modeling
- **Game Theory Analysis** - Strategic equilibrium analysis and learning dynamics
- **Bayesian Reasoning** - Replace "I think" with probabilistic evidence-based analysis
- **Chaos Detection** - Identify black swan events and early warning signals

### Technical Excellence
- **100% Test Coverage** - Comprehensive testing with pytest and hypothesis
- **MCP Integration** - Seamless Cursor AI integration via Model Context Protocol

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/codesmirnov/simulacrum-mcp.git
   cd simulacrum-mcp
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install the package:**
   ```bash
   pip install -e .
   ```

### Cursor AI Integration

1. **Configure MCP in Cursor:**
   Add to your Cursor settings (`.cursorrules` or global settings):

   ```json
   {
     "mcp": {
       "servers": {
         "simulacrum": {
           "command": "python",
           "args": ["-m", "simulacrum.server"],
           "env": {}
         }
       }
     }
   }
   ```

2. **Verify installation:**
   ```bash
   simulacrum-server --help
   ```

## 📖 Usage Examples

### System Dynamics Simulation

```python
from simulacrum import DynamicsSimulator

# Define a predator-prey model
scenario = {
    "name": "Lotka-Volterra Predator-Prey",
    "variables": [
        {"name": "prey", "initial_value": 10.0, "min_value": 0},
        {"name": "predator", "initial_value": 5.0, "min_value": 0}
    ],
    "equations": [
        {
            "target_variable": "prey",
            "expression": "prey * (2.0 - 0.01 * predator)"
        },
        {
            "target_variable": "predator",
            "expression": "predator * (-1.0 + 0.01 * prey)"
        }
    ],
    "config": {
        "time_config": {
            "end_time": 50.0,
            "time_step": 0.1
        }
    }
}

simulator = DynamicsSimulator()
result = simulator.simulate_dynamics(scenario)
print(f"Simulation completed: {result['status']}")
```

### Bayesian Belief Update

```python
from simulacrum import ProbabilityAnalyzer

analyzer = ProbabilityAnalyzer()

# Update beliefs with evidence
analysis = {
    "analysis_type": "bayesian_update",
    "prior": {"rain": 0.3, "no_rain": 0.7},
    "likelihood": {
        "cloudy": {"rain": 0.8, "no_rain": 0.4}
    },
    "evidence": [
        {"hypothesis": "rain", "observation": "cloudy", "strength": 1.0}
    ]
}

result = analyzer.analyze_probability(analysis)
print(f"Updated belief in rain: {result['final_posterior']['rain']:.2%}")
```

### Chaos Analysis

```python
from simulacrum import ChaosAnalyzer

analyzer = ChaosAnalyzer()

# Analyze time series for chaotic behavior
time_series_data = {
    "time_series": [0.1, 0.15, 0.08, 0.12, 0.18, 0.14, 0.09, 0.16, ...]
}

result = analyzer.analyze_chaos(time_series_data)
print(f"System state: {result['overall_assessment']['system_state']}")
print(f"Risk level: {result['overall_assessment']['risk_level']}")
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Install test dependencies
pip install -e ".[dev]"

# Run all tests with coverage
pytest --cov=simulacrum --cov-report=html

# Run specific test categories
pytest tests/test_dynamics.py
pytest tests/test_probability.py
pytest tests/test_chaos.py
```

## 🏗️ Architecture

### Core Principles
- **Single Responsibility** - Each class has one clear purpose
- **Open/Closed** - Extensible without modifying existing code
- **Liskov Substitution** - Subtypes are substitutable for base types
- **Interface Segregation** - Clients depend only on methods they use
- **Dependency Inversion** - Depend on abstractions, not concretions

### Package Structure
```
simulacrum/
├── core/           # Core engine and interfaces
├── tools/          # Analysis tools and simulators
├── validation/     # Data validation and type safety
└── server.py       # MCP server implementation
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Clone and setup
git clone https://github.com/codesmirnov/simulacrum-mcp.git
cd simulacrum-mcp

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
black simulacrum/
isort simulacrum/
mypy simulacrum/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built for the AI safety and alignment research community
- Inspired by system dynamics, chaos theory, and Bayesian epistemology
- Designed for practical deployment in AI assistant systems

## 📞 Contact

**codesmirnov**
- GitHub: [@codesmirnov](https://github.com/codesmirnov)
- Email: hello@codesmirnov.ru

---

**Transforming AI from pattern recognition to reality simulation.**
