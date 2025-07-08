cmake_minimum_required(VERSION 3.15)
project(fdcomp LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Build the core C++ library
add_library(fdcomp_cpp STATIC
    backend/cpp/src/trvl.cpp
)

target_include_directories(fdcomp_cpp
    PUBLIC backend/cpp/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(fdcomp_cpp PUBLIC ${OpenCV_LIBS})

# Build the pybind11 extension
pybind11_add_module(fdc_bindings
    backend/bindings/bindings.cpp
)

# Include necessary headers
target_include_directories(fdc_bindings PRIVATE
    backend/cpp/include
)

target_link_libraries(fdc_bindings PRIVATE
    fdcomp_cpp
)

# Set properties for the Python module
set_target_properties(fdc_bindings PROPERTIES
    CXX_VISIBILITY_PRESET "hidden"
    VISIBILITY_INLINES_HIDDEN YES
)

# Configure installation - install the compiled module into the fdcomp package directory
install(TARGETS fdc_bindings DESTINATION fdcomp)
