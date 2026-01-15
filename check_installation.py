#!/usr/bin/env python3
"""
Installation and functionality check script for Simulacrum MCP.
"""

import sys
import traceback
from typing import Dict, Any


def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version >= (3, 8):
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+")
        return False


def check_imports():
    """Check if all required packages can be imported."""
    print("\n📦 Checking imports...")

    required_packages = [
        ("numpy", "numpy"),
        ("pydantic", "pydantic"),
        ("scipy", "scipy"),
        ("mcp", "mcp"),
    ]

    all_good = True

    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
            print(f"✅ {package_name} - OK")
        except ImportError:
            print(f"❌ {package_name} - MISSING")
            print(f"   Install with: pip install {package_name}")
            all_good = False

    return all_good


def check_simulacrum_import():
    """Check if simulacrum package can be imported."""
    print("\n🔧 Checking Simulacrum package...")

    try:
        import simulacrum
        print(f"✅ Simulacrum {simulacrum.__version__} - OK")
        return True
    except ImportError as e:
        print(f"❌ Simulacrum import failed: {e}")
        print("   Make sure you're in the correct directory or package is installed")
        return False


def test_basic_functionality():
    """Test basic functionality of core components."""
    print("\n🧪 Testing basic functionality...")

    try:
        from simulacrum.tools.dynamics import DynamicsSimulator
        from simulacrum.validation.errors import ValidationError

        # Test simple scenario
        simulator = DynamicsSimulator()

        test_scenario = {
            "name": "Installation Test",
            "variables": [
                {"name": "x", "initial_value": 1.0}
            ],
            "equations": [
                {"target_variable": "x", "expression": "-0.1 * x"}
            ],
            "config": {
                "time_config": {
                    "end_time": 1.0,
                    "time_step": 0.1
                }
            }
        }

        result = simulator.simulate_dynamics(test_scenario)

        if result["status"] == "success":
            print("✅ Dynamics simulation - OK")
            return True
        else:
            print(f"❌ Dynamics simulation failed: {result}")
            return False

    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        traceback.print_exc()
        return False


def test_mcp_server():
    """Test MCP server initialization."""
    print("\n🔌 Testing MCP server...")

    try:
        from simulacrum.server import SimulacrumMCPServer

        server = SimulacrumMCPServer()
        print("✅ MCP server initialization - OK")
        return True

    except Exception as e:
        print(f"❌ MCP server test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all checks."""
    print("🧿 Simulacrum MCP Installation Check")
    print("=" * 50)

    checks = [
        check_python_version,
        check_imports,
        check_simulacrum_import,
        test_basic_functionality,
        test_mcp_server,
    ]

    results = []
    for check in checks:
        results.append(check())

    print("\n" + "=" * 50)
    print("📊 Results Summary:")

    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"🎉 All checks passed! ({passed}/{total})")
        print("\n🚀 You're ready to use Simulacrum MCP!")
        print("\nTo start the MCP server:")
        print("  python -m simulacrum.server")
        print("\nOr use the command line tool:")
        print("  simulacrum-server")
        return 0
    else:
        print(f"⚠️  Some checks failed: {passed}/{total} passed")
        print("\n🔧 Please fix the issues above and try again.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
