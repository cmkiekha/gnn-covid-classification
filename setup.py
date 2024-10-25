from setuptools import setup, find_packages

setup(
    name="gnn-covid-classification",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'pyyaml'
    ]
)