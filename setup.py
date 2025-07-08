from skbuild import setup
from setuptools import find_packages
import sys

setup(
    name="fdcomp",
    version="0.1.0",
    description="Fast Depth Compression Library with TRVL and RVL algorithms",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    cmake_install_dir="fdcomp",
    cmake_args=[
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DPYTHON_EXECUTABLE={sys.executable}",
        "-DCMAKE_VERBOSE_MAKEFILE=ON",
    ],
    cmake_source_dir=".",
    include_package_data=True,
    install_requires=[
        "numpy>=1.15.0",
    ],
    setup_requires=[
        "pybind11>=2.5.0",
        "scikit-build>=0.10.0",
    ],
    python_requires=">=3.6",
    zip_safe=False,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: C++",
    ],
)