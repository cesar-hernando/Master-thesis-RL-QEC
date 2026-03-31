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
    "pymatching",
    "gymnasium",
    "plotly",
    "scipy",
]

extras_require = {
    "torch": [
        "torch>=2.9",
        "torchvision>=0.24",
    ]
}



info = {
    "name": "adaptiveQRL",
    "version": version,
    "author": "Cesar Hernando",
    "author_email": "chernandodelaf@tudelft.nl",
    "description": "Adaptive quantum error decoding under drift noise via Graph Reinforcement Learning",
    "long_description": open('README.md').read(),
    "long_description_content_type": "text/markdown",
    "url": "https://github.com/cesar-hernando/Master-thesis-RL-QEC",
    "license": "Apache 2.0",
    "provides": ["adaptiveQRL"],
    "install_requires": install_requires,
    "extras_require": extras_require,
    "packages": find_packages(where='src'),
    "package_dir": {'': 'src'},
    "keywords": ["QEC", "Quantum", "Reinforcement Learning"],
}

classifiers = [
    "Development Status :: 3 - Alpha",
    "Programming Language :: Python :: 3.11",
    "License :: OSI Approved :: Apache Software License",
    "Intended Audience :: Science/Research",
]

setup(classifiers=classifiers, **info)