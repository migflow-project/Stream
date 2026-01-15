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
set(PYTHON_API_SRC
  __init__.py
  numerics/__init__.py
  numerics/cg.py 
  numerics/csr.py
  numerics/jacobi.py
  numerics/linsys.py
  mesh/__init__.py
  mesh/mesh.py
  mesh/ashape2D.py
  mesh/ashape3D.py
)

foreach(f ${PYTHON_API_SRC})
    configure_file("${f}" "${PYTHON_BUILD_DIR}/${f}" @ONLY)
endforeach()

# The 'DESTINATION' should be a location that is included in the user's PYTHON_PATH.
install(
    DIRECTORY ${PYTHON_BUILD_DIR}/
    DESTINATION "${PYTHON_INSTALL_DIR}"
    PATTERN "*.py"
)

