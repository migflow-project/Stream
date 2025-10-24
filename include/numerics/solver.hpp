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

#ifdef __cplusplus
}
#endif

namespace stream::numerics {
    struct Solver {
        virtual ~Solver();

        // init the solver for n unknowns
        virtual void init(uint32_t n);

        // Solve the system of equation 
        virtual void solve(LinearSystem const * const system, AvaDeviceArray<fp_tt, int>::Ptr sol);
    };

    struct ConjugateGradient : public Solver {
        AvaDeviceArray<fp_tt, int>::Ptr d_r;       // residue
        AvaDeviceArray<fp_tt, int>::Ptr d_d;       // search direction
        AvaDeviceArray<fp_tt, int>::Ptr d_d1;      // matrix * search direction
        AvaDeviceArray<fp_tt, int>::Ptr d_s;       // preconditioned direction
        AvaDeviceArray<fp_tt, int>::Ptr d_dots;    // gpu alpha/beta values
        AvaDeviceArray<fp_tt, int>::Ptr d_tmp_dot; // temp memory for dot product
        AvaDeviceArray<fp_tt, int>::Ptr d_temp;    // temp memory 
        AvaHostArray<fp_tt>::Ptr h_dots;           // cpu alpha/beta values
        size_t temp_size;

        ConjugateGradient();
        ~ConjugateGradient() override;

        void init(uint32_t n) override;
        void solve(LinearSystem const * const system, AvaDeviceArray<fp_tt, int>::Ptr sol) override;
    };

} // namespace stream::numerics

#endif // __STREAM_SOLVER_HPP__
