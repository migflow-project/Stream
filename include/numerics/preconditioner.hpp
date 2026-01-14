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
#ifndef __STREAM_PRECONDITIONER_HPP__
#define __STREAM_PRECONDITIONER_HPP__

// Forward declarations
#include "ava_device_array.h"
#include "csr.hpp"
namespace stream::numerics {
    struct Preconditioner;
    struct PrecJacobi;
} // namespace stream::numerics

// C-compatible typedefs and interface
#ifdef __cplusplus
extern "C" {
#endif

    typedef stream::numerics::Preconditioner Preconditioner_st;
    typedef stream::numerics::PrecJacobi PrecJacobi_st;

    PrecJacobi_st* PrecJacobi_create(d_CSR const * const A);
    void PrecJacobi_destroy(PrecJacobi_st * prec);

#ifdef __cplusplus
}
#endif

namespace stream::numerics {

    struct Preconditioner {
        virtual ~Preconditioner() {};

        // Compute the preconditioner on the device based on the 
        // matrix of the linear system of equations Ax = b
        virtual void init_device(const DeviceCSR& A) {};

        // Solve the preconditioner system M(sol) = rhs
        // Overwrite @sol
        virtual void solve(AvaDeviceArray<fp_tt, int>::Ptr sol, const AvaDeviceArray<fp_tt, int>::Ptr rhs) const {};
    };

    struct PrecJacobi : public Preconditioner {
        AvaDeviceArray<fp_tt, int>::Ptr d_idiag;

        PrecJacobi() noexcept;

        // Compute the preconditioner on the device based on the 
        // matrix of the linear system of equations Ax = b
        void init_device(const DeviceCSR& A) override;

        // Solve the preconditioner system Mx = b
        // Overwrite @sol
        void solve(AvaDeviceArray<fp_tt, int>::Ptr x, const AvaDeviceArray<fp_tt, int>::Ptr b) const override;
    };

} // namespace stream::numerics

#endif // __STREAM_PRECONDITIONER_HPP__
