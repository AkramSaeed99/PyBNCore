cmake_minimum_required(VERSION 3.24)
project(bncore VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Default to Release build for SIMD auto-vectorization
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Find Python and nanobind
find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)

# Core Library
add_library(bncore_core STATIC
    src/graph.cpp
    src/dense_tensor.cpp
    src/factor.cpp
    src/junction_tree.cpp
    src/compiler.cpp
    src/workspace.cpp
    src/engine.cpp
    src/manager.cpp
)
target_include_directories(bncore_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Optimization flags — critical for SIMD auto-vectorization and hot-path performance
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(bncore_core PRIVATE
    -O3                    # Full optimization
    -DNDEBUG               # Disable asserts in release
    -march=native          # Use all CPU features (NEON on ARM, AVX2 on x86)
    -fopenmp-simd          # Enable OpenMP SIMD pragmas without full OpenMP
    -funroll-loops          # Unroll inner loops for SIMD width
    -ftree-vectorize        # Auto-vectorize loops
  )
endif()

# Python Extension
nanobind_add_module(pybncore_ext src_python/main.cpp)
target_link_libraries(pybncore_ext PRIVATE bncore_core)
set_target_properties(pybncore_ext PROPERTIES OUTPUT_NAME "_core")

# Installation
install(TARGETS pybncore_ext DESTINATION pybncore)
