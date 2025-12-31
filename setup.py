from setuptools import setup, find_packages

setup(
    name="snn-dsl",         # Name of your package
    version="2025.12.31",   # Current version
    author="Felix Wang",
    author_email="felwang@sandia.gov",
    description="Compositional Spiking Neural Network Domain Specific Language",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),            # Automatically find packages
    package_dir={"": "src"},
    install_requires=[                   # List your dependencies here
        "numpy",
        "networkx",
    ],
)
