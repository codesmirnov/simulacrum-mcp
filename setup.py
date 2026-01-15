#!/usr/bin/env python3
"""
Simulacrum MCP - Reality Engine for AI Systems

A comprehensive toolkit for simulating complex dynamics, enabling AI systems
to understand and predict real-world phenomena through mathematical modeling.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="simulacrum-mcp",
    version="1.0.0",
    author="codesmirnov",
    author_email="contact@codesmirnov.dev",
    description="Reality Engine MCP - Advanced simulation toolkit for AI systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/codesmirnov/simulacrum-mcp",
    packages=find_packages(exclude=["tests*", "docs*"]),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "mypy>=1.0.0",
            "flake8>=4.0.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "simulacrum-server=simulacrum.server:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
