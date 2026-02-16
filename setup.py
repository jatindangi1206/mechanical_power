from setuptools import setup, find_packages

setup(
    name="mechanical_power",
    version="1.0.0",
    description="Mechanical Power Personalisation for ICU Patients using Offline RL",
    author="Jatin",
    author_email="jatin@ashoka.edu.in",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "scikit-learn>=1.3.0",
        "xgboost>=1.7.0",
        "d3rlpy>=2.0.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "loguru>=0.7.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "api": ["fastapi>=0.100.0", "uvicorn>=0.22.0"],
        "viz": ["matplotlib>=3.7.0", "seaborn>=0.12.0", "plotly>=5.14.0"],
        "dev": ["pytest>=7.4.0", "pytest-cov>=4.1.0", "jupyter>=1.0.0"],
    },
)
