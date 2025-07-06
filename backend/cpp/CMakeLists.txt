find_package(OpenCV REQUIRED)

add_library(fdcomp_cpp
    src/trvl.cpp
)
target_include_directories(fdcomp_cpp
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(fdcomp_cpp PUBLIC ${OpenCV_LIBS})

set_target_properties(fdcomp_cpp
    PROPERTIES PUBLIC_HEADER include/trvl.h
)
