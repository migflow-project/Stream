#include "csr.hpp"
#include <cstring>

#include "ava_device_array.hpp"
#include "ava_host_array.hpp"

namespace stream::numerics {

    DeviceCSR::DeviceCSR() noexcept {
        n = 0;
        d_row = AvaDeviceArray<uint32_t, int>::create({0});
        d_col = AvaDeviceArray<uint32_t, int>::create({0});
        d_val = AvaDeviceArray<fp_tt, int>::create({0});
    }
    
    void DeviceCSR::from_host(const HostCSR& h_csr) noexcept {
        n = h_csr.n;
        d_row->resize({h_csr.h_row->size()});
        d_col->resize({h_csr.h_col->size()});
        d_val->resize({h_csr.h_val->size()});

        gpu_memcpy(d_row->data, h_csr.h_row->data(), h_csr.h_row->size()*sizeof(uint32_t), gpu_memcpy_host_to_device);
        gpu_memcpy(d_col->data, h_csr.h_col->data(), h_csr.h_col->size()*sizeof(uint32_t), gpu_memcpy_host_to_device);
        gpu_memcpy(d_val->data, h_csr.h_val->data(), h_csr.h_val->size()*sizeof(fp_tt), gpu_memcpy_host_to_device);
    }

    DeviceCSR::DeviceCSRView DeviceCSR::to_view(void) const noexcept {
        return {
            d_row->to_view<-1>(),
            d_col->to_view<-1>(),
            d_val->to_view<-1>()
        };
    }

    HostCSR::HostCSR() noexcept {
        n = 0;
        h_row = AvaHostArray<uint32_t, int>::create({0});
        h_col = AvaHostArray<uint32_t, int>::create({0});
        h_val = AvaHostArray<fp_tt, int>::create({0});
    }

    HostCSR::HostCSR(
            uint32_t _n,
            uint32_t const * const row, 
            uint32_t const * const col, 
            fp_tt const * const val) noexcept {

        n = _n;
        uint32_t nnz = row[n];

        h_row = AvaHostArray<uint32_t, int>::create({(int) (n+1)});
        h_col = AvaHostArray<uint32_t, int>::create({(int) nnz});
        h_val = AvaHostArray<fp_tt, int>::create({(int) nnz});

        std::memcpy(h_row->data(), row, (n+1)*sizeof(*row));
        std::memcpy(h_col->data(), col, nnz*sizeof(*col));
        std::memcpy(h_val->data(), val, nnz*sizeof(*val));
    }

    
    // Copy a DeviceCSR to Host
    HostCSR::HostCSR(const DeviceCSR& d_csr) noexcept {
        n = d_csr.n;
        h_row = AvaHostArray<uint32_t, int>::create({d_csr.d_row->size});
        h_col = AvaHostArray<uint32_t, int>::create({d_csr.d_col->size});
        h_val = AvaHostArray<fp_tt, int>::create({d_csr.d_val->size});

        from_device(d_csr);
    }

    void HostCSR::from_device(const DeviceCSR& d_csr) noexcept {
        n = d_csr.n;
        h_row->resize({d_csr.d_row->size});
        h_col->resize({d_csr.d_col->size});
        h_val->resize({d_csr.d_val->size});

        gpu_memcpy(h_row->data(), d_csr.d_row->data, d_csr.d_row->size*sizeof(uint32_t), gpu_memcpy_device_to_host);
        gpu_memcpy(h_col->data(), d_csr.d_col->data, d_csr.d_col->size*sizeof(uint32_t), gpu_memcpy_device_to_host);
        gpu_memcpy(h_val->data(), d_csr.d_val->data, d_csr.d_val->size*sizeof(fp_tt), gpu_memcpy_device_to_host);
    }

} // namespace stream::numerics

extern "C" {

d_CSR* d_csr_create(){
    return new d_CSR;
}

void d_csr_h2d(d_CSR * dcsr, h_CSR const * const hcsr) {
    dcsr->from_host(*hcsr);
}

void d_csr_destroy(d_CSR* A) {
    delete A;
};


h_CSR* h_csr_create(
        uint32_t n,
        uint32_t const * const row,
        uint32_t const * const col,
        fp_tt const * const val)  {
    return new h_CSR(n, row, col, val);
}

void h_csr_destroy(h_CSR* A) {
    delete A;
}

} // extern C
