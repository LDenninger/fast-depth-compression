find_package(pybind11 CONFIG REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

pybind11_add_module(fdc_bindings
    bindings.cpp
)

# Include Python headers and project headers for rvl/trvl
target_include_directories(fdc_bindings PRIVATE
    ${Python3_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/cpp/include
)

target_link_libraries(fdc_bindings PRIVATE
    fdcomp_cpp
    ${Python3_LIBRARIES}
)