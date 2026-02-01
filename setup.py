"""
sage_refiner Setup Script
==========================

Standalone context compression library for RAG systems.
"""

from setuptools import find_packages, setup

# Read long description from README
try:
    with open("README.md", encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "Intelligent context compression algorithms for RAG systems"

# Read version
VERSION = "0.1.0"

# Core dependencies
INSTALL_REQUIRES = [
    "torch>=2.0.0,<3.0.0",
    "transformers>=4.52.0,<4.56.0",
    "numpy>=1.24.0,<3.0.0",
    "pyyaml>=6.0,<7.0",
    "json-repair>=0.30.0,<1.0.0",
    "tqdm>=4.65.0",
]

# Development dependencies
EXTRAS_REQUIRE = {
    "vllm": [
        "vllm>=0.9.2,<0.13",
    ],
    "reranker": [
        "FlagEmbedding>=1.0.0",
    ],
    "benchmark": [
        # SAGE 生态组件将在 quickstart 中分阶段安装
        # 数据分析和可视化
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "pandas>=2.0.0",
        "datasets>=2.14.0",
        "pyarrow>=10.0.0",
    ],
    "full": [
        "vllm>=0.9.2,<0.13",
        "FlagEmbedding>=1.0.0",
    ],
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "ruff>=0.1.0",
        "mypy>=1.0.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "pandas>=2.0.0",
        "datasets>=2.14.0",
    ],
    "test": [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
    ],
}

setup(
    name="sage-refiner",
    version=VERSION,
    description="Intelligent context compression algorithms for RAG systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="SAGE Team",
    author_email="sage-dev@example.com",
    url="https://github.com/intellistream/sageRefiner",
    project_urls={
        "Bug Tracker": "https://github.com/intellistream/sageRefiner/issues",
        "Source Code": "https://github.com/intellistream/sageRefiner",
        "Documentation": "https://github.com/intellistream/SAGE/tree/main/docs",
    },
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    python_requires=">=3.11",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="rag llm compression context nlp ai ml",
    license="Apache-2.0",
    include_package_data=True,
    zip_safe=False,
)
