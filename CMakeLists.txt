cmake_minimum_required(VERSION 3.15)
project(fdcomp LANGUAGES CXX)

option(BUILD_PYTHON_BINDINGS "Build the Python bindings" ON)
option(BUILD_CPP_TESTS "Build the C++ tests" OFF)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)



# Build the core C++ library
add_library(fdcomp_cpp STATIC
    backend/cpp/src/trvl.cpp
)

target_include_directories(fdcomp_cpp
    PUBLIC backend/cpp/include
)

# Build the pybind11 extension
if (BUILD_PYTHON_BINDINGS)
    message(STATUS "Python bindings will be built.")

    # Find required packages
    find_package(pybind11 CONFIG REQUIRED)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

    message(STATUS "Python interpreter: ${Python_EXECUTABLE}")
    message(STATUS "Python version: ${Python_VERSION}")

    pybind11_add_module(fdc_bindings
        backend/bindings/bindings.cpp
    )

    # Include necessary headers
    target_include_directories(fdc_bindings PRIVATE
        backend/cpp/include
    )

    # Link libraries to the Python extension module
    target_link_libraries(fdc_bindings PRIVATE
        fdcomp_cpp
        pybind11::pybind11
        Python3::Python
    )

    # Set properties for the Python module
    set_target_properties(fdc_bindings PROPERTIES
        CXX_VISIBILITY_PRESET "hidden"
        VISIBILITY_INLINES_HIDDEN YES
    )

    # Configure installation - install the compiled module into the fdcomp package directory
    install(TARGETS fdc_bindings DESTINATION fdcomp)
else()
    message(STATUS "Python bindings will not be built. Set BUILD_PYTHON_BINDINGS to ON to enable.")
endif()



# Install library and headers
install(TARGETS fdcomp_cpp
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        PUBLIC_HEADER DESTINATION include/fdcomp)

install(DIRECTORY backend/cpp/include/ DESTINATION include/fdcomp
        FILES_MATCHING PATTERN "*.h")


if (BUILD_CPP_TESTS)
    find_package(OpenCV REQUIRED)
    message(STATUS "C++ tests will be built.")
    add_executable(test_rvl tests/test_rvl.cpp)
    target_link_libraries(test_rvl PRIVATE fdcomp_cpp ${OpenCV_LIBS})
    target_include_directories(test_rvl PRIVATE ${OpenCV_INCLUDE_DIRS})

    add_executable(test_trvl tests/test_trvl.cpp)
    target_link_libraries(test_trvl PRIVATE fdcomp_cpp ${OpenCV_LIBS})
    target_include_directories(test_trvl PRIVATE ${OpenCV_INCLUDE_DIRS})

endif()
