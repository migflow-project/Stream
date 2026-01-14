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
#ifndef __STREAM_CSR_HPP__
#define __STREAM_CSR_HPP__

/*
 * Define a device CSR matrix and its host mirror.
 * The goal of these abstractions is to simplify the management of user systems. 
 *
 * e.g: 
 *  - The user supply a HostCSR that is then copied and solved on GPU.
 *  - The DeviceCSR is copied on CPU for debugging/comparison with CPU solver
 * 
 */

#include <cstdint>
#include "ava_device_array.h"
#include "ava_host_array.h"
#include "defines.h"

namespace stream::numerics {
    struct DeviceCSR;
    struct HostCSR;
} // namespace stream::numerics

#ifdef __cplusplus
extern "C" {
#endif

    typedef stream::numerics::DeviceCSR  d_CSR;
    typedef stream::numerics::HostCSR    h_CSR;

    // Create an empty device csr
    d_CSR* d_csr_create();

    // Destroy a device csr
    void d_csr_destroy(d_CSR* A);

    // Create a host csr from arrays
    h_CSR* h_csr_create(
        uint32_t n,
        uint32_t const * const row,
        uint32_t const * const col,
        fp_tt const * const val
    );

    // Destroy a host csr
    void h_csr_destroy(h_CSR* A);

    // Copy a host csr to a device csr
    void d_csr_h2d(d_CSR * dcsr, h_CSR const * const hcsr);

#ifdef __cplusplus
}
#endif

namespace stream::numerics {

    struct DeviceCSR {

        uint32_t n;
        AvaDeviceArray<uint32_t, int>::Ptr d_row;  // compressed row indices
        AvaDeviceArray<uint32_t, int>::Ptr d_col;  // columns
        AvaDeviceArray<fp_tt, int>::Ptr    d_val;  // matrix n x n 
                                                      
        DeviceCSR() noexcept;
        
        void from_host(const HostCSR& h_csr) noexcept;

        struct DeviceCSRView {
            AvaView<uint32_t, -1> d_row_v; 
            AvaView<uint32_t, -1> d_col_v; 
            AvaView<fp_tt, -1>    d_val_v; 
        };

        DeviceCSRView to_view(void) const noexcept;
    };

    struct HostCSR {
        uint32_t n;
        AvaHostArray<uint32_t, int>::Ptr h_row;  // compressed row indices
        AvaHostArray<uint32_t, int>::Ptr h_col;  // columns
        AvaHostArray<fp_tt, int>::Ptr    h_val;     // matrix n x n 

        // Default
        HostCSR() noexcept;

        // Init from raw arrays
        HostCSR(
            uint32_t n,
            uint32_t const * const row,
            uint32_t const * const col,
            fp_tt const * const val
        ) noexcept;

        // Init from device
        HostCSR(const DeviceCSR& d_csr) noexcept;

        // Overwrite from device
        void from_device(const DeviceCSR& d_csr) noexcept; 
    };
} // namespace stream::numerics

#endif // __STREAM_CSR_HPP__
