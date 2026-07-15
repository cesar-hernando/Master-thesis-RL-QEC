from setuptools import setup, find_packages

# Read the version from the VERSION file
def read_version():
    with open('VERSION', 'r') as version_file:
        return version_file.read().strip()

version = 0.1 #read_version()

install_requires = [

    "numpy",
    "matplotlib",
    "stim",
    # Custom fork: adds a regularized reweight strength `alpha` on top of
    # PyMatching's own correlated matching (enable_correlations=True), @69d7c049.
    # NOT stock PyPI — install it first
    # from the fork per README section 3. Left UNPINNED here on purpose so
    # `pip install -e .` accepts the pre-installed fork wheel instead of forcing
    # a from-git source rebuild (which fails on Windows due to path length).
    "pymatching",
    "gymnasium",
    "plotly",
    "scipy",
    "torch",
    "torch_geometric",
]

extras_require = {
    "torch": [
        "torch>=2.9",
        "torchvision>=0.24",
    ]
}



info = {
    "name": "NeuralCM",
    "version": version,
    "author": "Cesar Hernando",
    "author_email": "chernandodelaf@tudelft.nl",
    "description": "Adaptive quantum error decoding under drift noise via Graph Reinforcement Learning",
    "long_description": open('README.md').read(),
    "long_description_content_type": "text/markdown",
    "url": "https://github.com/cesar-hernando/Master-thesis-RL-QEC",
    "license": "Apache 2.0",
    "provides": ["NeuralCM"],
    "install_requires": install_requires,
    "extras_require": extras_require,
    "packages": find_packages(where='src'),
    "package_dir": {'': 'src'},
    "keywords": ["QEC", "Surface Code", "Reinforcement Learning"],
}

classifiers = [
    "Development Status :: 3 - Alpha",
    "Programming Language :: Python :: 3.11",
    "License :: OSI Approved :: Apache Software License",
    "Intended Audience :: Science/Research",
]

setup(classifiers=classifiers, **info)