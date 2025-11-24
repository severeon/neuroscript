#!/usr/bin/env python
"""Setup script for NeuroScript v2."""

from setuptools import setup, find_packages

setup(
    name="neuroscript",
    version="2.0.0",
    description="Type-safe neural architecture composition system with capability-based shape inference",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
)
