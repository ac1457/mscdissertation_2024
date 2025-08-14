#!/usr/bin/env python3
"""
Setup script for Two-Stage Lending Model Package
================================================

This script makes the project installable as a Python package.

Usage:
    pip install -e .          # Install in development mode
    pip install .             # Install in production mode
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Two-Stage Lending Model with Sentiment Analysis"

# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="two-stage-lending-model",
    version="1.0.0",
    author="AI Assistant",
    author_email="ai.assistant@example.com",
    description="Advanced loan default prediction with two-stage modeling and sentiment analysis",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    keywords=["machine learning", "lending", "default prediction", "sentiment analysis", "ensemble methods"],
    url="https://github.com/yourusername/two-stage-lending-model",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "pytest>=6.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "lending-model=main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
) 