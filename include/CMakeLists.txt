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
target_include_directories(utils
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/utils/>
           $<INSTALL_INTERFACE:include/stream/utils/>
)

if (TARGET geometry)
    target_include_directories(geometry
        PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/geometry/>
               $<INSTALL_INTERFACE:include/stream/geometry/>
    )

    target_include_directories(geometry
        PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/>
               $<INSTALL_INTERFACE:include/stream/>
    )

    add_subdirectory(geometry/)
endif()

if (TARGET numerics)
    target_include_directories(numerics
        PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/numerics/>
               $<INSTALL_INTERFACE:include/stream/numerics/>
    )

    target_include_directories(numerics
        PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/>
               $<INSTALL_INTERFACE:include/stream/>
    )

    add_subdirectory(numerics/)
endif()

if (TARGET mesh)
    target_include_directories(mesh
        PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/mesh/>
               $<INSTALL_INTERFACE:include/stream/mesh/>
    )

    target_include_directories(mesh
        PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/>
               $<INSTALL_INTERFACE:include/stream/>
    )

    add_subdirectory(mesh/)
endif()
