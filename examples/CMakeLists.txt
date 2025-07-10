# Examples CMakeLists.txt

# MNIST Classification Example
add_executable(mnist_classification mnist_classification.cpp)
target_link_libraries(mnist_classification neuralforge)
target_compile_features(mnist_classification PRIVATE cxx_std_17)

# Set optimization flags for examples
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        target_compile_options(mnist_classification PRIVATE -O3 -march=native -mavx2)
    endif()
endif()

# Regression Example
add_executable(regression_example regression_example.cpp)
target_link_libraries(regression_example neuralforge)
target_compile_features(regression_example PRIVATE cxx_std_17)

# Simple Demo
add_executable(simple_demo simple_demo.cpp)
target_link_libraries(simple_demo neuralforge)
target_compile_features(simple_demo PRIVATE cxx_std_17)

# Install examples (optional)
if(NEURALFORGE_INSTALL_EXAMPLES)
    install(TARGETS mnist_classification regression_example simple_demo
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/neuralforge-examples
    )
endif() 