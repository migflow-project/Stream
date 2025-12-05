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
option(ENABLE_CUDA "Compile with CUDA" ON)
set(ENABLE_ARCH "native" CACHE STRING "Target architecture. E.g. 89 for CUDA compilation or armv8-a for CPU compilation. Default=native")

if (ENABLE_CUDA)
    message(STATUS "Stream compile Target : CUDA. Architecture : ${ENABLE_ARCH}")
    enable_language(CUDA)

    set(AVA_TARGET "CUDA" CACHE STRING "Compilation target")
    add_compile_definitions(AVA_TARGET_CUDA)
else()
    message(STATUS "Stream compile Target : CPU")
    set(AVA_TARGET "CPU" CACHE STRING "Compilation target")
    add_compile_definitions(AVA_TARGET_CPU)
endif()

include(./deps/ava/ava.cmake)

if (ENABLE_CUDA)
    set(CMAKE_CUDA_FLAGS "--extended-lambda --extra-device-vectorization -g -lineinfo")
    set(CMAKE_CUDA_ARCHITECTURES ${ENABLE_ARCH})
endif()
# ============================= Define targets ================================

set(STREAM_MODULES
    geometry
    numerics
    mesh
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
    endforeach()
    if (ENABLE_CUDA)
        set_property(TARGET ${M} PROPERTY CUDA_ARCHITECTURES ${ENABLE_ARCH})
    endif()
endforeach()


target_link_libraries(mesh PUBLIC geometry)

# ============================= Add directories ===============================
add_subdirectory(include/)       # Library includes and interface
add_subdirectory(src/)           # Source code
add_subdirectory(python/)        # Python bindings

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

# List files and properties for build pipeline debug
function(list_dir_without_cmake_src DIR_LIST)
    foreach(DIR ${DIR_LIST})
        string(REPLACE ${CMAKE_SOURCE_DIR} "" CLEAN_STRING ${DIR})
        message("\t ${CLEAN_STRING}")
    endforeach()
endfunction()

get_target_property(geometry_INCLUDE_DIRS geometry INCLUDE_DIRECTORIES )
message(STATUS "Stream::geometry headers dirs (public + private + interface):")
list_dir_without_cmake_src("${geometry_INCLUDE_DIRS}")

# ======================= Print install directories ===========================

message(STATUS "Library install directory : ${CMAKE_INSTALL_LIBDIR}")
message(STATUS "Python bindings install directory : ${PYTHON_INSTALL_DIR}")
message(STATUS "Python bindings build directory : ${PYTHON_BUILD_DIR}")

