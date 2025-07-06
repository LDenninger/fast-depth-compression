find_package(pybind11 CONFIG REQUIRED)            # via pip or system install :contentReference[oaicite:2]{index=2}

pybind11_add_module(fdc_bindings
    bindings.cpp
)
target_link_libraries(fdc_bindings PRIVATE fdcomp_cpp)