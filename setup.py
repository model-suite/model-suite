from setuptools import setup, find_packages

setup(
    name="model-suite",
    packages=find_packages(exclude=[]),
    version="0.0.4",
    license="Apache License 2.0",
    description="Model Suite",
    url="https://github.com/model-suite/model-suite",
    install_requires=["torch"],
)
