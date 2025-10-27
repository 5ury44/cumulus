"""
Setup script for Cumulus SDK
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="cumulus-sdk",
    version="1.0.0",
    author="Cumulus Team",
    author_email="team@cumulus.ai",
    description="Cumulus-based distributed execution SDK",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cumulus-ai/cumulus",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: System :: Distributed Computing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
    ],
    python_requires=">=3.8",
    install_requires=[
        "requests>=2.31.0",
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "python-multipart>=0.0.6",
        "pydantic>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "ml": [
            "torch>=1.12.0",
            "numpy>=1.21.0",
            "scikit-learn>=1.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "cumulus-cli=cumulus.cli:main",
            "cumulus-worker=cumulus.start_worker:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
