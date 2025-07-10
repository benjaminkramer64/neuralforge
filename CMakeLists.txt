cmake_minimum_required(VERSION 3.16)

project(NeuralForge 
    VERSION 1.0.0
    DESCRIPTION "High-Performance Neural Network Library for C++"
    LANGUAGES CXX
)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -mtune=native")

# Enable SIMD optimizations
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mavx2 -mfma")
endif()

# Options
option(NEURALFORGE_BUILD_EXAMPLES "Build examples" ON)
option(NEURALFORGE_BUILD_TESTS "Build tests" ON)
option(NEURALFORGE_BUILD_BENCHMARKS "Build benchmarks" ON)
option(NEURALFORGE_ENABLE_OPENMP "Enable OpenMP support" ON)
option(NEURALFORGE_ENABLE_CUDA "Enable CUDA support" OFF)

# Find packages
find_package(Threads REQUIRED)

# OpenMP support
if(NEURALFORGE_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
    if(OpenMP_CXX_FOUND)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        message(STATUS "OpenMP support enabled")
    endif()
endif()

# CUDA support (future feature)
if(NEURALFORGE_ENABLE_CUDA)
    find_package(CUDA QUIET)
    if(CUDA_FOUND)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        message(STATUS "CUDA support enabled")
        add_definitions(-DNEURALFORGE_CUDA_ENABLED)
    else()
        message(WARNING "CUDA requested but not found")
        set(NEURALFORGE_ENABLE_CUDA OFF)
    endif()
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Header-only library target
add_library(neuralforge INTERFACE)
target_include_directories(neuralforge INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(neuralforge INTERFACE cxx_std_17)

# Link libraries
target_link_libraries(neuralforge INTERFACE Threads::Threads)

if(NEURALFORGE_ENABLE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(neuralforge INTERFACE OpenMP::OpenMP_CXX)
endif()

# Platform-specific optimizations
if(WIN32)
    target_compile_definitions(neuralforge INTERFACE NEURALFORGE_WINDOWS)
elseif(APPLE)
    target_compile_definitions(neuralforge INTERFACE NEURALFORGE_MACOS)
    # Use Accelerate framework on macOS
    find_library(ACCELERATE_FRAMEWORK Accelerate)
    if(ACCELERATE_FRAMEWORK)
        target_link_libraries(neuralforge INTERFACE ${ACCELERATE_FRAMEWORK})
        target_compile_definitions(neuralforge INTERFACE NEURALFORGE_USE_ACCELERATE)
    endif()
else()
    target_compile_definitions(neuralforge INTERFACE NEURALFORGE_LINUX)
endif()

# Detect CPU features
include(CheckCXXSourceRuns)

set(CMAKE_REQUIRED_FLAGS "-mavx2")
check_cxx_source_runs("
    #include <immintrin.h>
    int main() {
        __m256 a = _mm256_set1_ps(1.0f);
        __m256 b = _mm256_set1_ps(2.0f);
        __m256 c = _mm256_add_ps(a, b);
        return 0;
    }
" HAVE_AVX2)

if(HAVE_AVX2)
    target_compile_definitions(neuralforge INTERFACE NEURALFORGE_HAS_AVX2)
    message(STATUS "AVX2 support detected and enabled")
endif()

set(CMAKE_REQUIRED_FLAGS "-mavx512f")
check_cxx_source_runs("
    #include <immintrin.h>
    int main() {
        __m512 a = _mm512_set1_ps(1.0f);
        __m512 b = _mm512_set1_ps(2.0f);
        __m512 c = _mm512_add_ps(a, b);
        return 0;
    }
" HAVE_AVX512)

if(HAVE_AVX512)
    target_compile_definitions(neuralforge INTERFACE NEURALFORGE_HAS_AVX512)
    message(STATUS "AVX-512 support detected and enabled")
endif()

# Examples
if(NEURALFORGE_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Tests
if(NEURALFORGE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Benchmarks
if(NEURALFORGE_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# Install configuration
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Install headers
install(DIRECTORY include/neuralforge
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp"
)

# Install library target
install(TARGETS neuralforge
    EXPORT NeuralForgeTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install export targets
install(EXPORT NeuralForgeTargets
    FILE NeuralForgeTargets.cmake
    NAMESPACE NeuralForge::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/NeuralForge
)

# Create config file
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/NeuralForgeConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/NeuralForgeConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/NeuralForge
)

# Create version file
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/NeuralForgeConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

# Install config files
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/NeuralForgeConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/NeuralForgeConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/NeuralForge
)

# Generate pkg-config file
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/neuralforge.pc.in"
    "${CMAKE_CURRENT_BINARY_DIR}/neuralforge.pc"
    @ONLY
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/neuralforge.pc"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
)

# Print configuration summary
message(STATUS "")
message(STATUS "NeuralForge Configuration Summary:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "  Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")
message(STATUS "Features:")
message(STATUS "  Examples: ${NEURALFORGE_BUILD_EXAMPLES}")
message(STATUS "  Tests: ${NEURALFORGE_BUILD_TESTS}")
message(STATUS "  Benchmarks: ${NEURALFORGE_BUILD_BENCHMARKS}")
message(STATUS "  OpenMP: ${NEURALFORGE_ENABLE_OPENMP}")
message(STATUS "  CUDA: ${NEURALFORGE_ENABLE_CUDA}")
message(STATUS "  AVX2: ${HAVE_AVX2}")
message(STATUS "  AVX-512: ${HAVE_AVX512}")
message(STATUS "")

# Uninstall target
if(NOT TARGET uninstall)
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
        IMMEDIATE @ONLY
    )

    add_custom_target(uninstall
        COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake
    )
endif()

# Documentation target (if Doxygen is available)
find_package(Doxygen QUIET)
if(DOXYGEN_FOUND)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in
                   ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
    
    add_custom_target(docs
        ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM
    )
endif()

# CPack configuration for packaging
set(CPACK_PACKAGE_NAME "NeuralForge")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "High-Performance Neural Network Library for C++")
set(CPACK_PACKAGE_VENDOR "NeuralForge Development Team")
set(CPACK_PACKAGE_CONTACT "developer@neuralforge.ai")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")

# Platform-specific packaging
if(WIN32)
    set(CPACK_GENERATOR "ZIP;NSIS")
    set(CPACK_NSIS_DISPLAY_NAME "NeuralForge ${PROJECT_VERSION}")
    set(CPACK_NSIS_PACKAGE_NAME "NeuralForge")
elseif(APPLE)
    set(CPACK_GENERATOR "TGZ;DragNDrop")
else()
    set(CPACK_GENERATOR "TGZ;DEB;RPM")
    
    # Debian package configuration
    set(CPACK_DEBIAN_PACKAGE_MAINTAINER "NeuralForge Team <developer@neuralforge.ai>")
    set(CPACK_DEBIAN_PACKAGE_DEPENDS "libc6 (>= 2.27)")
    set(CPACK_DEBIAN_PACKAGE_SECTION "devel")
    
    # RPM package configuration
    set(CPACK_RPM_PACKAGE_GROUP "Development/Libraries")
    set(CPACK_RPM_PACKAGE_LICENSE "MIT")
endif()

include(CPack)

# Performance testing macro
macro(add_performance_test name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} neuralforge)
    target_compile_definitions(${name} PRIVATE NEURALFORGE_PERFORMANCE_TEST)
    
    # Add as a test that runs with timing
    add_test(NAME ${name}_performance
        COMMAND ${CMAKE_COMMAND} -E time $<TARGET_FILE:${name}>
    )
endmacro()

# Code coverage target (GCC/Clang only)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    option(NEURALFORGE_ENABLE_COVERAGE "Enable code coverage" OFF)
    
    if(NEURALFORGE_ENABLE_COVERAGE)
        target_compile_options(neuralforge INTERFACE --coverage)
        target_link_libraries(neuralforge INTERFACE --coverage)
        
        find_program(GCOV_PATH gcov)
        find_program(LCOV_PATH lcov)
        find_program(GENHTML_PATH genhtml)
        
        if(GCOV_PATH AND LCOV_PATH AND GENHTML_PATH)
            add_custom_target(coverage
                COMMAND ${LCOV_PATH} --directory . --capture --output-file coverage.info
                COMMAND ${LCOV_PATH} --remove coverage.info '/usr/*' --output-file coverage.info
                COMMAND ${LCOV_PATH} --list coverage.info
                COMMAND ${GENHTML_PATH} -o coverage coverage.info
                WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                COMMENT "Generating code coverage report"
            )
            
            message(STATUS "Code coverage enabled")
        endif()
    endif()
endif() 