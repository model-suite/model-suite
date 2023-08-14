from setuptools import setup, find_packages

setup(
    name="model-suite",
    packages=find_packages(exclude=[]),
    version="0.0.1",
    description="Model Suite",
    url="https://github.com/model-suite/model-suite",
    install_requires=["torch"],
)