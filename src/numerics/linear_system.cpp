#include "ava_device_array.hpp"
#include "csr.hpp"
#include "linear_system.hpp"
#include "defines.h"

extern "C" {

LinSys* LinSys_create(void) {
    LinSys* ret = new LinSys;
    ret->n = 0;
    ret->d_b = AvaDeviceArray<fp_tt, int>::create({0});
    return ret;
}

void LinSys_destroy(LinSys *sys) {
    delete sys;
}

void LinSys_set(LinSys* sys, const d_CSR *const A, const fp_tt *const b){
    sys->n = A->n;
    sys->d_csr = *A;
    sys->d_b->resize({(int) A->n});
    gpu_memcpy(sys->d_b->data, b, A->n*sizeof(*b), gpu_memcpy_host_to_device);
}

} // extern C
