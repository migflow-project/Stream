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

#ifdef __cplusplus
}
#endif

namespace stream::numerics {

    struct DeviceCSR {
        AvaDeviceArray<uint32_t, int>::Ptr d_row;  // compressed row indices
        AvaDeviceArray<uint32_t, int>::Ptr d_col;  // columns
        AvaDeviceArray<fp_tt, int>::Ptr    d_val;     // matrix n x n 
                                                      
        DeviceCSR() noexcept;
        
        // Copy a HostCSR to Device
        DeviceCSR(const HostCSR& h_csr) noexcept;

        void from_host(const HostCSR& h_csr) noexcept;
    };

    struct HostCSR {
        AvaHostArray<uint32_t, int>::Ptr h_row;  // compressed row indices
        AvaHostArray<uint32_t, int>::Ptr h_col;  // columns
        AvaHostArray<fp_tt, int>::Ptr    h_val;     // matrix n x n 

        HostCSR() noexcept;
        // Copy a DeviceCSR to Host
        HostCSR(const DeviceCSR& d_csr) noexcept;
        void from_device(const DeviceCSR& d_csr) noexcept; 
    };
} // namespace stream::numerics

#endif // __STREAM_CSR_HPP__
