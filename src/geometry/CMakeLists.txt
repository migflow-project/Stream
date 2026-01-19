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

set(GEOMETRY_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/vec.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/bbox.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/primitives.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lbvh.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lbvh_agglo.cpp
)
target_sources(geometry PRIVATE ${GEOMETRY_SRC})

foreach(file ${GEOMETRY_SRC})
    set_source_files_properties(${file} TARGET_DIRECTORY geometry PROPERTIES LANGUAGE ${AVA_LANGUAGE})
endforeach()
