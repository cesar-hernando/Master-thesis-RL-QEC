from setuptools import setup, find_packages
import pathlib

# Read the version from the VERSION file
def read_version():
    with open('VERSION', 'r') as version_file:
        return version_file.read().strip()

version = "0.1.0"

install_requires = [
    "numpy>=1.21",
    "scipy>=1.8",
    "networkx>=2.8",
    "stim>=1.9",
    "pymatching>=0.16",
    "plotly>=5.0",
    "torch>=1.13",
]

extras_require = {
    "dev": [
        "ipykernel",
        "matplotlib",
        "pytest",
        "black",
    ],
    "torch_geom": [
        "torch-scatter",
        "torch-sparse",
        "torch-cluster",
        "torch-spline-conv",
        "torch-geometric",
    ],
}



info = {
    "name": "master-thesis-rl-qec",
    "version": version,
    "author": "Cesar Hernando",
    "author_email": "",
    "description": "Adaptive MWPM reweighting with a GNN+SAC agent (master thesis code)",
    "long_description": open('README.md').read() if (pathlib.Path('README.md')).exists() else "",
    "long_description_content_type": "text/markdown",
    "url": "https://github.com/cesar-hernando/Master-thesis-RL-QEC",
    "license": "MIT",
    "packages": find_packages(),
    "python_requires": ">=3.8",
    "install_requires": install_requires,
    "extras_require": extras_require,
    "include_package_data": True,
}

classifiers = [
    "Development Status :: 3 - Alpha",
    "Programming Language :: Python :: 3.9",
    "License :: OSI Approved :: Apache Software License",
    "Intended Audience :: Science/Research",
]

setup(classifiers=classifiers, **info)