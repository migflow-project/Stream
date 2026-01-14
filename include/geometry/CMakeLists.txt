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

# ===================== geometry ===========================

# Set header when installing target 
target_sources(geometry PUBLIC
    FILE_SET geometry_public_headers
    TYPE HEADERS
    BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/vec.hpp
          ${CMAKE_CURRENT_SOURCE_DIR}/bbox.hpp
          ${CMAKE_CURRENT_SOURCE_DIR}/primitives.hpp
          ${CMAKE_CURRENT_SOURCE_DIR}/morton.hpp
          ${CMAKE_CURRENT_SOURCE_DIR}/lbvh.hpp
          ${CMAKE_CURRENT_SOURCE_DIR}/trbvh.hpp
          ${CMAKE_CURRENT_SOURCE_DIR}/lbvh_agglo.hpp
          ${CMAKE_CURRENT_SOURCE_DIR}/predicates.hpp
          ${CMAKE_CURRENT_SOURCE_DIR}/traversal_stack.hpp
)
