cmake_minimum_required(VERSION 3.26 FATAL_ERROR)

Project(Stream CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
set(BUILD_SHARED_LIBS ON)

## Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'RelWithDebInfo' as none was specified.")
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Extra compile flags
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_compile_options(-Wall -Wextra -Wshadow -Wno-unknown-pragmas)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options(-Wall -Wextra -Wno-unused-function -Wno-unused-parameter -Wno-unknown-pragmas -g -lineinfo)
endif()


# ============================= Define options ================================

option(ENABLE_TESTS "Compile tests" OFF)

option(ENABLE_CUDA "Compile with CUDA" ON)
set(ENABLE_CUDA_ARCH "native" CACHE STRING "CUDA architecture")

include(./deps/ava/ava.cmake)

if (ENABLE_CUDA)
    message(STATUS "Stream compile Target : CUDA. Architecture : ${ENABLE_CUDA_ARCH}")
    enable_language(CUDA)

    set(CMAKE_CUDA_FLAGS "--extended-lambda --extra-device-vectorization")
    set(CMAKE_CUDA_ARCHITECTURES ${ENABLE_CUDA_ARCH})

    set(AVA_TARGET "CUDA" CACHE STRING "Compilation target")
    add_compile_definitions(AVA_TARGET_CUDA)
else()
    message(STATUS "Target : CPU")
    set(AVA_TARGET "CPU" CACHE STRING "Compilation target")
    add_compile_definitions(AVA_TARGET_CPU)
endif()

# ============================= Define targets ================================

add_library(geometry SHARED)
add_library(Stream::geometry ALIAS geometry)
target_include_directories(geometry PUBLIC "${AVA_INCLUDE_DIRS}")

add_library(numerics SHARED)
add_library(Stream::numerics ALIAS numerics)
target_include_directories(numerics PUBLIC "${AVA_INCLUDE_DIRS}")

if (ENABLE_CUDA)
    set_property(TARGET geometry PROPERTY CUDA_ARCHITECTURES ${ENABLE_CUDA_ARCH})
    set_property(TARGET numerics PROPERTY CUDA_ARCHITECTURES ${ENABLE_CUDA_ARCH})
endif()

# ============================= Add directories ===============================

add_subdirectory(include/)
add_subdirectory(src/)

if (ENABLE_TESTS)
    add_subdirectory(tests/)
endif()

# ========================= Print properties of files =========================

get_target_property(geometry_COMPILE_OPTIONS geometry COMPILE_OPTIONS )
message(STATUS "Stream::geometry compile options : ${geometry_COMPILE_OPTIONS}")

message(STATUS "Stream::geometry sources and properties:")
get_target_property(geometry_SOURCES geometry SOURCES)
foreach(F ${geometry_SOURCES})
    get_source_file_property(PROPS ${F} LANGUAGE)
    string(REPLACE ${CMAKE_SOURCE_DIR} "" CLEAN_STRING ${F})
    message("\t ${CLEAN_STRING} : ${PROPS}")
endforeach()
