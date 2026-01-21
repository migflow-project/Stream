# 
# Stream - Copyright (C) <2025-2026>
# <Universite catholique de Louvain (UCL), Belgique>
# 
# List of the contributors to the development of Stream: see AUTHORS file.
# Description and complete License: see LICENSE file.
# 
# This file is part of Stream. Stream is free software:
# you can redistribute it and/or modify it under the terms of the GNU Lesser General
# Public License as published by the Free Software Foundation, either version 3
# of the License, or (at your option) any later version.
# 
# Stream is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Lesser General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public License along with Stream. 
# If not, see <https://www.gnu.org/licenses/>.
# 
cmake_minimum_required(VERSION 3.26 FATAL_ERROR)

Project(Stream CXX)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(BUILD_SHARED_LIBS ON)

## Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'RelWithDebInfo' as none was specified.")
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Extra compile flags
add_compile_options(
    -Wall 
    -Wextra 
    -Wnon-virtual-dtor
    -Wno-unknown-pragmas
    -Wduplicated-cond
    -Wduplicated-branches
    -Wcast-align
)
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_compile_options(-Wshadow)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options(-Wno-unused-function -Wno-unused-parameter)
endif()


# ============================= Define options ================================

set(LIB_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}" CACHE STRING "Library install directory")
set(PYTHON_BUILD_DIR "${PROJECT_BINARY_DIR}/stream" CACHE STRING "Python bindings build directory")
set(PYTHON_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/stream" CACHE STRING "Python bindings install directory")


option(ENABLE_TESTS "Compile tests" OFF)
option(ENABLE_CUDA "Compile with CUDA" OFF)
option(ENABLE_HIP "Compile with HIP" OFF)
option(ENABLE_TRIANGLE_REDUNDANCY "When set, every node will compute ALL incident simplices" ON)

set(ENABLE_ARCH "native" CACHE STRING "Target architecture. E.g. 89 for CUDA compilation or armv8-a for CPU compilation. Default=native")

if (ENABLE_CUDA)
    message(STATUS "Stream compile Target : CUDA. Architecture : ${ENABLE_ARCH}")
    set(AVA_TARGET "CUDA" CACHE STRING "Compilation target")
elseif (ENABLE_HIP)
    message(STATUS "Stream compile Target : HIP. Architecture : ${ENABLE_ARCH}")
    set(AVA_TARGET "HIP" CACHE STRING "Compilation target")
else()
    message(STATUS "Stream compile Target : CPU. Architecture : ${ENABLE_ARCH}")
    set(AVA_TARGET "CPU" CACHE STRING "Compilation target")
endif()

include(./deps/ava/ava.cmake)

# ============================= Define targets ================================

set(STREAM_MODULES
    geometry
    numerics
    mesh
    utils
)

message(STATUS "Module list : ${STREAM_MODULES}")

foreach(M ${STREAM_MODULES})
    add_library(${M} SHARED)
    add_library(Stream::${M} ALIAS ${M})
    foreach(D ${AVA_INCLUDE_DIRS})
        target_include_directories(${M} 
            PUBLIC $<BUILD_INTERFACE:${D}>
                   $<INSTALL_INTERFACE:include/ava/>
        )

        target_compile_options(${M} PUBLIC ${AVA_COMPILE_OPTIONS})
    endforeach()
endforeach()

target_link_libraries(utils PUBLIC)
target_link_libraries(geometry PUBLIC utils ${AVA_LINK_LIBRARIES})
target_link_libraries(mesh PUBLIC geometry utils ${AVA_LINK_LIBRARIES})
target_link_libraries(numerics PUBLIC utils ${AVA_LINK_LIBRARIES})

if (ENABLE_TRIANGLE_REDUNDANCY) 
    target_compile_definitions(mesh PRIVATE TRIANGLE_REDUNDANCY)
endif()

# ============================= Add directories ===============================
add_subdirectory(include/)       # Library includes and interface
add_subdirectory(src/)           # Source code
add_subdirectory(python/)        # Python bindings
add_subdirectory(testcases)      # Testcases

if (ENABLE_TESTS)
    add_subdirectory(tests/)     # Tests
endif()

# ============================ Install targets ================================

foreach(M ${STREAM_MODULES})
    if (TARGET ${M})
        install(
            TARGETS ${M}
            EXPORT StreamTargets
            FILE_SET ${M}_public_headers 
        )
    endif()
endforeach()

# ======================= Print install directories ===========================

message(STATUS "Library install directory : ${CMAKE_INSTALL_LIBDIR}")
message(STATUS "Python bindings install directory : ${PYTHON_INSTALL_DIR}")
message(STATUS "Python bindings build directory : ${PYTHON_BUILD_DIR}")

# ======================= Install targets =====================================

install(
    EXPORT StreamTargets
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Stream
    NAMESPACE Stream::
    FILE Stream.cmake # Not sure if this is still needed
)

configure_package_config_file( 
    "Config.cmake.in" 
    "StreamConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Stream
    PATH_VARS
    CMAKE_INSTALL_LIBDIR
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/StreamConfigVersion.cmake
    VERSION 0.0.0
    COMPATIBILITY SameMajorVersion
)

### Install Config and ConfigVersion files
install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/StreamConfig.cmake"
          "${CMAKE_CURRENT_BINARY_DIR}/StreamConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Stream"
)
