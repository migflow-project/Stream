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
#ifndef __STREAM_LINEAR_SYSTEM_HPP__
#define __STREAM_LINEAR_SYSTEM_HPP__

/*
 * Define a generic linear system Ax = b where :
 *      - A is a sparse matrix stored in CSR format (rowptr, col, val)
 *      - x is a dense solution vector
 *      - b is the independant dense vector
 */

#include <cstdint>
#include "ava_device_array.h"
#include "defines.h"
#include "csr.hpp"

// forward definitions
namespace stream::numerics {
    struct LinearSystem;
} // namespace stream::numerics

// C compatible typedefs
#ifdef __cplusplus
extern "C" {
#endif

    typedef stream::numerics::LinearSystem  LinSys;

    // Create an empty linear system
    LinSys* LinSys_create(void);

    // Set the matrix A and independant vector b of the system
    void LinSys_set(LinSys * sys, d_CSR const * const A, fp_tt const * const b);

    // Destroy the linear system
    void LinSys_destroy(LinSys* sys);

#ifdef __cplusplus
}
#endif

namespace stream::numerics {
    struct LinearSystem {
        uint32_t n;      // Size of the system
        DeviceCSR d_csr; // CSR Matrix
        AvaDeviceArray<fp_tt, int>::Ptr d_b; // Independant vector
    };
} // namespace stream::numerics

#endif // __STREAM_LINEAR_SYSTEM_HPP__
