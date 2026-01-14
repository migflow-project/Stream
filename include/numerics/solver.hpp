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
#ifndef __STREAM_SOLVER_HPP__
#define __STREAM_SOLVER_HPP__

/*
 *  Define an abstract solver to solve linear systems given 
 *  in the form of LinearSystem structures.
 *  This structure has to be subclassed to implements or interface 
 *  algorithms, e.g.:
 *      - Conjugate Gradient 
 *      - cuDSS
 *
 */

// Forward definitions
#include <cstdint>
#include "defines.h"
#include "linear_system.hpp"
#include "ava_device_array.h"
#include "preconditioner.hpp"

namespace stream::numerics {
    struct Solver;
    struct ConjugateGradient;
} // namespace stream::numerics

// C compatible typedef
#ifdef __cplusplus
extern "C" {
#endif

    typedef stream::numerics::Solver Solver_st;
    typedef stream::numerics::ConjugateGradient SolverCG;

    // Create an empty solver
    SolverCG* SolverCG_create();
    // Destroy a solver
    void SolverCG_destroy(SolverCG* solver);

    // Solve a system of equations using Conjugate Gradient with Jacobi preconditioner
    uint32_t SolverCG_jacobi_solve(SolverCG * solver, LinSys const * const sys, PrecJacobi_st const * const prec, fp_tt * x);

    uint32_t SolverCG_solve(SolverCG * solver, LinSys const * const sys, fp_tt * x);

#ifdef __cplusplus
}
#endif

namespace stream::numerics {
    struct Solver {
        virtual ~Solver() {};
        // Solve the system of equation 
        virtual uint32_t solve_precond(LinearSystem const * const system, Preconditioner const * const prec, AvaDeviceArray<fp_tt, int>::Ptr sol) { 
            return 0;
        };
        virtual uint32_t solve(LinearSystem const * const system, AvaDeviceArray<fp_tt, int>::Ptr sol) { 
            return 0;
        };
    };

    struct ConjugateGradient : public Solver {
        AvaDeviceArray<fp_tt, int>::Ptr d_r;       // residue
        AvaDeviceArray<fp_tt, int>::Ptr d_d;       // search direction
        AvaDeviceArray<fp_tt, int>::Ptr d_d1;      // matrix * search direction
        AvaDeviceArray<fp_tt, int>::Ptr d_s;       // preconditioned direction
        AvaDeviceArray<fp_tt, int>::Ptr d_dots;    // gpu dot products
        AvaDeviceArray<fp_tt, int>::Ptr d_tmp_dot; // temp memory for dot product
        AvaDeviceArray<fp_tt, int>::Ptr d_temp;    // temp memory 
        size_t temp_size;

        ConjugateGradient();
        ~ConjugateGradient();

        fp_tt dot_prod(AvaDeviceArray<fp_tt, int>::Ptr d_u, AvaDeviceArray<fp_tt, int>::Ptr d_v) noexcept;

        // Solve the system without preconditioner
        uint32_t solve(LinearSystem const * const system, AvaDeviceArray<fp_tt, int>::Ptr x) override;

        // Solve the system using the generic preconditioner @prec
        uint32_t solve_precond(LinearSystem const * const system, Preconditioner const * const prec, AvaDeviceArray<fp_tt, int>::Ptr x) override;
    };

} // namespace stream::numerics

#endif // __STREAM_SOLVER_HPP__
