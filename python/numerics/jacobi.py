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
from .. import libnum
from .csr import d_csr_ptr, DeviceCSR

import ctypes

# ========================== Bindings to C functions =========================

prec_jacobi_ptr = ctypes.c_void_p

PrecJacobi_create = libnum.PrecJacobi_create
PrecJacobi_create.restype = prec_jacobi_ptr
PrecJacobi_create.argtypes = [d_csr_ptr]

PrecJacobi_destroy = libnum.PrecJacobi_destroy
PrecJacobi_destroy.restype = None
PrecJacobi_destroy.argtypes = [prec_jacobi_ptr]


# ========================== Python wrapper classes =========================

class PrecJacobi:
    def __init__(self, dcsr: DeviceCSR):
        self.prec = PrecJacobi_create(dcsr.dcsr)

    def __del__(self):
        PrecJacobi_destroy(self.prec)
