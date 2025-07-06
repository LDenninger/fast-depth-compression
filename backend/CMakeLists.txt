cmake_minimum_required(VERSION 3.15)
project(myproj LANGUAGES CXX)

# Build the core C++ library
add_subdirectory(cpp)

# Build the pybind11 extension
add_subdirectory(bindings)