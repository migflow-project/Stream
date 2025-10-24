#include "csr.hpp"

#include "ava_device_array.hpp"
#include "ava_host_array.hpp"

namespace stream::numerics {

    DeviceCSR::DeviceCSR() noexcept {
        d_row = AvaDeviceArray<uint32_t, int>::create({0});
        d_col = AvaDeviceArray<uint32_t, int>::create({0});
        d_val = AvaDeviceArray<fp_tt, int>::create({0});
    }
    
    // Copy a HostCSR to Device
    DeviceCSR::DeviceCSR(const HostCSR& h_csr) noexcept {
        d_row = AvaDeviceArray<uint32_t, int>::create({h_csr.h_row->size()});
        d_col = AvaDeviceArray<uint32_t, int>::create({h_csr.h_col->size()});
        d_val = AvaDeviceArray<fp_tt, int>::create({h_csr.h_val->size()});

        from_host(h_csr);
    }

    void DeviceCSR::from_host(const HostCSR& h_csr) noexcept {
        d_row->resize({h_csr.h_row->size()});
        d_col->resize({h_csr.h_col->size()});
        d_val->resize({h_csr.h_val->size()});

        gpu_memcpy(d_row->data, h_csr.h_row->data(), h_csr.h_row->size()*sizeof(uint32_t), gpu_memcpy_host_to_device);
        gpu_memcpy(d_col->data, h_csr.h_col->data(), h_csr.h_col->size()*sizeof(uint32_t), gpu_memcpy_host_to_device);
        gpu_memcpy(d_val->data, h_csr.h_val->data(), h_csr.h_val->size()*sizeof(fp_tt), gpu_memcpy_host_to_device);
    }

    HostCSR::HostCSR() noexcept {
        h_row = AvaHostArray<uint32_t, int>::create({0});
        h_col = AvaHostArray<uint32_t, int>::create({0});
        h_val = AvaHostArray<fp_tt, int>::create({0});
    }
    
    // Copy a DeviceCSR to Host
    HostCSR::HostCSR(const DeviceCSR& d_csr) noexcept {
        h_row = AvaHostArray<uint32_t, int>::create({d_csr.d_row->size});
        h_col = AvaHostArray<uint32_t, int>::create({d_csr.d_col->size});
        h_val = AvaHostArray<fp_tt, int>::create({d_csr.d_val->size});

        from_device(d_csr);
    }

    void HostCSR::from_device(const DeviceCSR& d_csr) noexcept {
        h_row->resize({d_csr.d_row->size});
        h_col->resize({d_csr.d_col->size});
        h_val->resize({d_csr.d_val->size});

        gpu_memcpy(h_row->data(), d_csr.d_row->data, d_csr.d_row->size*sizeof(uint32_t), gpu_memcpy_device_to_host);
        gpu_memcpy(h_col->data(), d_csr.d_col->data, d_csr.d_col->size*sizeof(uint32_t), gpu_memcpy_device_to_host);
        gpu_memcpy(h_val->data(), d_csr.d_val->data, d_csr.d_val->size*sizeof(fp_tt), gpu_memcpy_device_to_host);
    }

} // namespace stream::numerics

