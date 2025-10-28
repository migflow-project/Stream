#include "ava_device_array.h"
#include "ava_view.h"
#include "csr.hpp"
#include "defines.h"
#include "ava_device_array.hpp"
#include "preconditioner.hpp"

namespace stream::numerics {

    PrecJacobi::PrecJacobi() noexcept {
        d_idiag = AvaDeviceArray<fp_tt, int>::create({0});
    }

    void PrecJacobi::init_device(const DeviceCSR& A) {
        d_idiag->resize({(int) A.n});
        AvaView<fp_tt, -1> d_idiag_v = d_idiag->to_view<-1>(); 
        DeviceCSR::DeviceCSRView A_v = A.to_view();

        // Find the diagonal of the CSR and store its inverse
        ava_for<256>(nullptr, 0, A.n, [=] __device__ (uint32_t const i){
            uint32_t const start = A_v.d_row_v(i);
            uint32_t const end = A_v.d_row_v(i+1);

            for (uint32_t j = start; j < end; j++) {
                if (A_v.d_col_v(j) == i) {
                    d_idiag_v(i) = 1.0f/A_v.d_val_v(j);
                    return;
                }
            }
        });
    }

    void PrecJacobi::solve(AvaDeviceArray<fp_tt, int>::Ptr x, const AvaDeviceArray<fp_tt, int>::Ptr b) const {
        AvaView<fp_tt, -1> d_idiag_v = d_idiag->to_view<-1>(); 
        AvaView<fp_tt, -1> d_x_v = x->to_view<-1>(); 
        AvaView<fp_tt, -1> d_b_v = b->to_view<-1>(); 

        ava_for<256>(nullptr, 0, x->size, [=] __device__ (uint32_t const i){
                d_x_v(i) = d_b_v(i) * d_idiag_v(i);
        });
    }

} // namespace stream::numerics

extern "C" {

PrecJacobi_st* PrecJacobi_create(d_CSR const * const A) {
    PrecJacobi_st * ret = new PrecJacobi_st;
    ret->init_device(*A);
    return ret;
}

void PrecJacobi_destroy(PrecJacobi_st * prec) {
    delete prec;
}

} // extern C
