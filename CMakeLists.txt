# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2026 Max-Planck-Institut für Kernphysik
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at
# https://mozilla.org/MPL/2.0/.

cmake_minimum_required(VERSION 3.15...3.30)

# Derive the version from the git tag (see cmake/GitVersion.cmake). This must run before
# project(), which needs a numeric MAJOR.MINOR.PATCH; PHEPEX_VERSION_STRING carries the full
# `git describe` string for display (the C++ PHEPEX_VERSION macro).
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(GitVersion)

project(phepex VERSION ${PHEPEX_VERSION} LANGUAGES CXX
        DESCRIPTION "Photo-electron pulse extraction kernels")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Generate the public version header from the git-derived version into the build tree.
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/include/phepex/version.hpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/include/phepex/version.hpp"
    @ONLY)

# When built as a Python wheel (scikit-build-core defines SKBUILD), build the bindings and
# skip the C++ dev-install (headers/lib/cmake/pc) that only matters for C++ consumers.
if(DEFINED SKBUILD)
  set(_phepex_python_default ON)
else()
  set(_phepex_python_default OFF)
endif()
option(PHEPEX_BUILD_PYTHON "Build the nanobind Python module (phepex._core)" ${_phepex_python_default})
option(PHEPEX_BUILD_EXAMPLES "Build the C++ example" OFF)
option(PHEPEX_BUILD_TESTS "Build the standalone C++ unit tests (vendored Catch2)" OFF)
option(PHEPEX_BUILD_BENCHMARKS "Build the C++ per-kernel micro-benchmark" OFF)

# neighbor_peak_indices accumulates neighbour waveforms in pairs by default, which halves
# the accumulator read-modify-write traffic (~17-22% faster on the benchmarked aarch64
# core). The extra branch and altered vectorisation may not pay off on every target, so
# the strategy is selectable; OFF falls back to the sequential sum.
option(PHEPEX_NEIGHBOR_PAIRWISE_SUM
       "Use pairwise neighbour-sum accumulation in neighbor_peak_indices" ON)

# Row-tile width for the batched preprocess_waveforms path: the number of waveforms processed
# per tile, one per SIMD lane.
set(PHEPEX_PREPROCESS_TILE_WIDTH 24 CACHE STRING
    "SIMD-lane tile width for the batched preprocess_waveforms path")

# libphepex: built as both shared (libphepex.so) and static (libphepex.a)
set(PHEPEX_SOURCES
    src/preprocess.cpp
    src/clip.cpp
    src/neighbor.cpp
    src/extract.cpp
    src/generate.cpp)

add_library(phepex SHARED ${PHEPEX_SOURCES})
add_library(phepex_static STATIC ${PHEPEX_SOURCES})
add_library(phepex::phepex ALIAS phepex)
add_library(phepex::phepex_static ALIAS phepex_static)

foreach(tgt phepex phepex_static)
  target_include_directories(${tgt} PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
  target_compile_features(${tgt} PUBLIC cxx_std_17)
  target_compile_options(${tgt} PRIVATE -O3)
  target_compile_definitions(${tgt} PRIVATE
      PHEPEX_NEIGHBOR_PAIRWISE_SUM=$<IF:$<BOOL:${PHEPEX_NEIGHBOR_PAIRWISE_SUM}>,1,0>
      PHEPEX_PREPROCESS_TILE_WIDTH=${PHEPEX_PREPROCESS_TILE_WIDTH})
  set_target_properties(${tgt} PROPERTIES
      OUTPUT_NAME phepex
      POSITION_INDEPENDENT_CODE ON)
endforeach()
set_target_properties(phepex PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR})

# Python bindings (phepex._core), statically linking libphepex into the module
if(PHEPEX_BUILD_PYTHON)
  # ${SKBUILD_SABI_COMPONENT} is set by scikit-build-core to "Development.SABIModule" when
  # building a stable-ABI (abi3) wheel and is empty otherwise. It must be requested here so
  # FindPython creates the Python::SABIModule target that nanobind's STABLE_ABI build links
  # against; without it nanobind silently falls back to a version-specific module while the
  # wheel is still tagged abi3 (a broken, mis-tagged wheel).
  find_package(Python 3.10 REQUIRED
               COMPONENTS Interpreter Development.Module ${SKBUILD_SABI_COMPONENT})
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import nanobind; print(nanobind.cmake_dir())"
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
  find_package(nanobind CONFIG REQUIRED)
  nanobind_add_module(_core STABLE_ABI NB_STATIC src/bindings.cpp)
  target_link_libraries(_core PRIVATE phepex_static)
  install(TARGETS _core LIBRARY DESTINATION phepex)
endif()

# C++ example (optional)
if(PHEPEX_BUILD_EXAMPLES)
  add_executable(phepex_example examples/example.cpp)
  target_link_libraries(phepex_example PRIVATE phepex::phepex)
endif()

# Standalone C++ unit tests (vendored Catch2 v3)
if(PHEPEX_BUILD_TESTS)
  enable_testing()
  add_executable(phepex_tests
      tests/cpp/test_phepex.cpp
      third_party/catch2/catch_amalgamated.cpp)
  target_include_directories(phepex_tests PRIVATE third_party/catch2)
  target_link_libraries(phepex_tests PRIVATE phepex::phepex)
  add_test(NAME phepex_tests COMMAND phepex_tests)
endif()

# C++ per-kernel micro-benchmark (optional). Links the shared library and is built at -O3
# regardless of CMAKE_BUILD_TYPE so its absolute timings are meaningful. The camera
# configuration it exercises is read at run time from benchmarks/flashcam-config.txt
# (default path relative to the repository root).
if(PHEPEX_BUILD_BENCHMARKS)
  add_executable(phepex-microbench benchmarks/cpp/microbench.cpp)
  target_link_libraries(phepex-microbench PRIVATE phepex::phepex)
  target_compile_options(phepex-microbench PRIVATE -O3)
endif()

# Install the C++ library (headers, targets, CMake package config, pkg-config)
if(NOT DEFINED SKBUILD)
  install(TARGETS phepex phepex_static EXPORT phepexTargets
          LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
          ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
          RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
  install(DIRECTORY include/phepex DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
          PATTERN "version.hpp.in" EXCLUDE)
  # version.hpp is generated (git-derived); install the generated copy, not the template.
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/include/phepex/version.hpp"
          DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/phepex")

  install(EXPORT phepexTargets NAMESPACE phepex::
          DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/phepex)
  configure_package_config_file(
      cmake/phepexConfig.cmake.in
      "${CMAKE_CURRENT_BINARY_DIR}/phepexConfig.cmake"
      INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/phepex)
  write_basic_package_version_file(
      "${CMAKE_CURRENT_BINARY_DIR}/phepexConfigVersion.cmake"
      VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
  install(FILES
          "${CMAKE_CURRENT_BINARY_DIR}/phepexConfig.cmake"
          "${CMAKE_CURRENT_BINARY_DIR}/phepexConfigVersion.cmake"
          DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/phepex)

  # pkg-config for Makefile-based consumers
  configure_file(phepex.pc.in "${CMAKE_CURRENT_BINARY_DIR}/phepex.pc" @ONLY)
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/phepex.pc"
          DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
endif()
