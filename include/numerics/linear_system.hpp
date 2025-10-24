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

#ifdef __cplusplus
}
#endif

namespace stream::numerics {
    struct LinearSystem {
        uint32_t n;  // Size of the system
                     
        // CSR Matrix
        DeviceCSR d_csr;

        // Independant vector
        AvaDeviceArray<fp_tt, int>::Ptr d_b;  // independant vector n x k

        virtual ~LinearSystem();
    };
} // namespace stream::numerics

#endif // __STREAM_LINEAR_SYSTEM_HPP__
