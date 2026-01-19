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
set(IMR26_TESTS
    uniform2D
    uniform3D
    ashape_from_file2D
)

foreach(T ${IMR26_TESTS})
    add_executable(${T} "${T}.cpp")
    target_link_libraries(${T} PRIVATE mesh)
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/${T}.cpp PROPERTIES LANGUAGE ${AVA_LANGUAGE})
endforeach()

set(PYTHON_IMR26_TESTS
    run_on_simulation_meshes.py
)

foreach(f ${PYTHON_IMR26_TESTS})
    configure_file("${f}" "${CMAKE_CURRENT_BINARY_DIR}/${f}" @ONLY)
endforeach()
