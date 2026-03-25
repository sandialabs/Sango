from setuptools import setup, find_packages

setup(
    name="sango",
    version="1.0.0",
    author="Felix Wang",
    author_email="felwang@sandia.gov",
    description="Compositional Spiking Neural Network Domain Specific Language",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "networkx",
    ],
)
