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

# ===================== mesh ===========================

# Set header when installing target 
target_sources(mesh PUBLIC
    FILE_SET mesh_public_headers
    TYPE HEADERS
    BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/mesh.hpp
          ${CMAKE_CURRENT_SOURCE_DIR}/DirectAlphaShape2D.hpp
          ${CMAKE_CURRENT_SOURCE_DIR}/DirectAlphaShape3D.hpp
)
