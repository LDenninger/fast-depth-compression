# python/setup.py
from skbuild import setup
from setuptools import find_packages

setup(
    name="fdcomp",
    version="0.1.0",
    packages=find_packages(),
    cmake_install_dir="fdcomp",
    cmake_args=[
        "-DBUILD_PYTHON_BINDINGS=ON",  # Enable Python bindings
        "-DCMAKE_BUILD_TYPE=Release"   # Set build type to Release
    ],
    install_requires=[
        "numpy",  # Add any dependencies required by the package
    ],
    python_requires=">=3.6",  # Specify the minimum Python version
)