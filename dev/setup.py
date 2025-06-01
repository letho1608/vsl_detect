#!/usr/bin/env python3
"""
Setup script for Vietnamese Sign Language Detection System
"""

from setuptools import setup, find_packages
import os

# Read README
def read_readme():
    with open('README.md', 'r', encoding='utf-8') as f:
        return f.read()

# Read requirements
def read_requirements():
    with open('requirements.txt', 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="vsl-detect",
    version="1.0.0",
    author="Vietnamese Sign Language Team",
    author_email="letho1608@example.com",
    description="Real-time Vietnamese Sign Language Recognition System",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/letho1608/vsl_detect",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Video :: Capture",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
            "pre-commit>=2.20.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.18.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "vsl-detect=vsl_detect.main:main",
            "vsl-train=vsl_detect.data.trainer:main",
            "vsl-collect=vsl_detect.data.collector:main",
            "vsl-process=vsl_detect.data.processor:main",
        ],
    },
    include_package_data=True,
    package_data={
        "vsl_detect": [
            "configs/*.yaml",
            "configs/*.json",
            "ui/icons/*.png",
            "ui/styles/*.qss",
        ],
    },
    zip_safe=False,
    keywords="sign-language, deep-learning, computer-vision, mediapipe, tensorflow, vietnamese",
    project_urls={
        "Bug Reports": "https://github.com/letho1608/vsl_detect/issues",
        "Source": "https://github.com/letho1608/vsl_detect",
        "Documentation": "https://vsl-detect.readthedocs.io/",
    },
)