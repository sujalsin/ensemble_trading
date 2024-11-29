from setuptools import setup, find_packages

setup(
    name="ensemble_trading",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "torch",
        "yfinance",
        "ta",
        "gymnasium"
    ]
)
