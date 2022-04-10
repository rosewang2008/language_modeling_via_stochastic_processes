from distutils.core import setup
from setuptools import find_packages

setup(
    name='language_modeling_via_stochastic_processes',
    version='0.1dev',
    packages=['language_modeling_via_stochastic_processes',],
    install_requires=[
        "dotmap==1.3.23",
        "datasets=2.0.0",
        "hydra-core==1.1.1",
        "matplotlib==3.3.4",
        "numpy==1.19.2",
        "pandas==1.2.3",
        "pytorch_lightning==1.5.8",
        "scikit_learn==1.0.2",
        "scipy==1.6.2",
        "seaborn==0.11.1",
        "sentence_transformers==2.0.0",
        "six==1.15.0",
        "tensorflow==2.4.1",
        "torch",
        "torchvision",
        "tqdm==4.49.0",
        "wandb==0.10.23",
        "numpy",
        "Pillow",
        "pyparsing",
        "six",
        "scipy",
        "pytz",
        "tqdm",
        "packaging",
    ]
    )