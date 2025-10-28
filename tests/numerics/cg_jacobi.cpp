#include <cstdint>
#include <vector>

#include "ava.h"
#include "ava_device_array.h"
#include "ava_device_array.hpp"
#include "ava_host_array.h"
#include "ava_host_array.hpp"
#include "numerics/linear_system.hpp"
#include "numerics/preconditioner.hpp"
#include "numerics/csr.hpp"
#include "numerics/solver.hpp"


int main(void) {

    uint32_t n = 100;

    // Create a diagonal matrix of size n
    //    with diagonal 1, 2, 3, 4, 5, ...
    // And an independant vector of ones
    std::vector<uint32_t> row(n+1);
    std::vector<uint32_t> col(n);
    std::vector<float> val(n);
    std::vector<float> b(n);
    std::vector<float> x(n);


    row[0] = 0;
    for (uint32_t i = 0; i < n; i++) {
        row[i+1] = i+1;
        col[i] = i;
        val[i] = static_cast<float>(i+1);
        b[i] = 1.0f;
    }

    h_CSR h_csr(n, row.data(), col.data(), val.data());
    d_CSR d_csr;
    d_csr.from_host(h_csr);

    AvaDeviceArray<fp_tt, int>::Ptr d_b = AvaDeviceArray<fp_tt, int>::create({(int) n});
    AvaDeviceArray<fp_tt, int>::Ptr d_x = AvaDeviceArray<fp_tt, int>::create({(int) n});
    gpu_memcpy(d_b->data, b.data(), n*sizeof(float), gpu_memcpy_host_to_device);

    LinSys sys;
    sys.n = n;
    sys.d_b = d_b;
    sys.d_csr = d_csr;

    PrecJacobi_st prec;
    prec.init_device(d_csr);

    SolverCG cg;
    cg.init(n);

    uint32_t niter = cg.solve(&sys, &prec, d_x);
    gpu_device_synchronise();
    gpu_memcpy(x.data(), d_x->data, n*sizeof(float), gpu_memcpy_device_to_host);
    
    for (uint32_t i = 0; i < n; i++) {
        if (std::fabs(x[i] - b[i]/(i+1)) > 1e-7f) {
            printf("x[%u] = %.5e, expected %.5e\n", i, x[i], b[i]/(i+1));
        }
    }
}
