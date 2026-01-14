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
target_sources(numerics
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/csr.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/conjugate_gradient.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/preconditioner_jacobi.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/linear_system.cpp
)

if (ENABLE_CUDA) 
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/csr.cpp TARGET_DIRECTORY numerics PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/conjugate_gradient.cpp TARGET_DIRECTORY numerics PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/preconditioner_jacobi.cpp TARGET_DIRECTORY numerics PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/linear_system.cpp TARGET_DIRECTORY numerics PROPERTIES LANGUAGE CUDA)
endif()
