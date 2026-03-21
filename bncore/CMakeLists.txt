cmake_minimum_required(VERSION 3.24)
project(bncore VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

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

# Python Extension
nanobind_add_module(pybncore_ext src_python/main.cpp)
target_link_libraries(pybncore_ext PRIVATE bncore_core)
set_target_properties(pybncore_ext PROPERTIES OUTPUT_NAME "_core")

# Installation
install(TARGETS pybncore_ext DESTINATION pybncore)
