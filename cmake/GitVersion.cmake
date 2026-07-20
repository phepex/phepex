# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2026 Max-Planck-Institut für Kernphysik
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at
# https://mozilla.org/MPL/2.0/.

# Resolve the project version from the git tag and expose two variables in the caller:
#   PHEPEX_VERSION_STRING  full descriptive version, e.g. "0.1.0" or "0.1.0-5-gabc1234-dirty"
#                          (used verbatim wherever a version *string* is wanted)
#   PHEPEX_VERSION         numeric MAJOR.MINOR.PATCH, suitable for project(VERSION ...)
#
# Resolution order:
#   1. SKBUILD_PROJECT_VERSION -- set by scikit-build-core (from setuptools-scm) during wheel
#      and sdist builds, so the compiled library matches the Python package version exactly,
#      including when there is no .git (building from an sdist).
#   2. `git describe --tags --dirty` in a git checkout.
#   3. "0.0.0" fallback (a bare source copy with neither git metadata nor a provided version).

function(phepex_resolve_version)
  set(_raw "")

  if(DEFINED SKBUILD_PROJECT_VERSION AND NOT SKBUILD_PROJECT_VERSION STREQUAL "")
    set(_raw "${SKBUILD_PROJECT_VERSION}")
  else()
    find_package(Git QUIET)
    if(GIT_FOUND)
      execute_process(
        COMMAND "${GIT_EXECUTABLE}" describe --tags --dirty --always
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        OUTPUT_VARIABLE _raw
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE _git_result)
      if(NOT _git_result EQUAL 0)
        set(_raw "")
      endif()
    endif()
  endif()

  if(_raw STREQUAL "")
    set(_raw "0.0.0")
    message(WARNING
      "phepex: could not determine the version from git or SKBUILD_PROJECT_VERSION; "
      "falling back to ${_raw}")
  endif()

  # Strip a leading "v" from git tags like v0.1.0.
  string(REGEX REPLACE "^v" "" _version_string "${_raw}")

  # Extract the numeric MAJOR.MINOR.PATCH prefix for project(VERSION ...); default missing parts to 0.
  if(_version_string MATCHES "^([0-9]+)\\.([0-9]+)\\.([0-9]+)")
    set(_numeric "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}.${CMAKE_MATCH_3}")
  elseif(_version_string MATCHES "^([0-9]+)\\.([0-9]+)")
    set(_numeric "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}.0")
  elseif(_version_string MATCHES "^([0-9]+)")
    set(_numeric "${CMAKE_MATCH_1}.0.0")
  else()
    set(_numeric "0.0.0")
  endif()

  set(PHEPEX_VERSION_STRING "${_version_string}" PARENT_SCOPE)
  set(PHEPEX_VERSION "${_numeric}" PARENT_SCOPE)

  # Re-run CMake when the checked-out commit changes so the version stays current.
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git/HEAD")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
                 "${CMAKE_CURRENT_SOURCE_DIR}/.git/HEAD")
  endif()
endfunction()

phepex_resolve_version()
