// 
// Stream - Copyright (C) <2025-2026>
// <Universite catholique de Louvain (UCL), Belgique>
// 
// List of the contributors to the development of Stream: see AUTHORS file.
// Description and complete License: see LICENSE file.
// 
// This file is part of Stream. Stream is free software:
// you can redistribute it and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.
// 
// Stream is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
// without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU Lesser General Public License for more details.
// 
// You should have received a copy of the GNU Lesser General Public License along with Stream. 
// If not, see <https://www.gnu.org/licenses/>.
// 
#include "ava_device_array.hpp"
#include "csr.hpp"
#include "linear_system.hpp"
#include "defines.h"

extern "C" {

LinSys* LinSys_create(void) {
    LinSys* ret = new LinSys;
    ret->n = 0;
    ret->d_b = AvaDeviceArray<fp_tt, int>::create({0});
    return ret;
}

void LinSys_destroy(LinSys *sys) {
    delete sys;
}

void LinSys_set(LinSys* sys, const d_CSR *const A, const fp_tt *const b){
    sys->n = A->n;
    sys->d_csr = *A;
    sys->d_b->resize({(int) A->n});
    deep_copy(sys->d_b->data, b, A->n);
}

} // extern C
