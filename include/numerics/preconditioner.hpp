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
