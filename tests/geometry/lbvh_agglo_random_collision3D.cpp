/*
 * Test LBVH from [Apetrei, 2014] on random spheres in 3D
 */
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <ctime>

#include "ava.h"
#include "ava_device_array.h"
#include "ava_host_array.h"
#include "ava_host_array.hpp"
#include "ava_view.h"
#include "ava_scan.h"

#include "vec.hpp"
#include "bbox.hpp"
#include "lbvh_agglo.hpp"
#include "primitives.hpp"


int main(int argc, char** argv){

    int n = 100;
    if (argc > 1) {
        n = (int) strtoll(argv[1], NULL, 10);
    }
 
    // ====== Generate points on a line, distant from 1 with radius 0.55 ======
    AvaHostArray<Sphere3D>::Ptr h_coords = AvaHostArray<Sphere3D>::create({n});
    float const min_r =  1.f / std::pow((float)n, 1.0f/3.0f);
    for (int i = 0; i < n; i++){
        Sphere3D s;
        s.c[0] = ((float) rand()) / RAND_MAX;
        s.c[1] = ((float) rand()) / RAND_MAX;
        s.c[2] = ((float) rand()) / RAND_MAX;
        s.r = 0.25f * min_r * (1.f + ((float) rand()) / RAND_MAX);
        h_coords(i) = s;
    }

    AvaDeviceArray<Sphere3D, int>::Ptr d_coords = AvaDeviceArray<Sphere3D, int>::create({n});
    d_coords->set(h_coords);

    // Construct lbvh
    Sphere3DLBVHa lbvh;
    lbvh.set_objects(d_coords);
    struct timespec t0, t1;
    gpu_device_synchronise();
    timespec_get(&t0, TIME_UTC);
    lbvh.build();
    gpu_device_synchronise();
    timespec_get(&t1, TIME_UTC);
    printf("Construction time : %f ms\n", 
            (t1.tv_sec - t0.tv_sec)*1e3 + (t1.tv_nsec - t0.tv_nsec)*1e-6);

    uint32_t root_id;
    gpu_memcpy(&root_id, lbvh.d_root->data, sizeof(uint32_t), gpu_memcpy_device_to_host);

    BBox3f root_data;
    gpu_memcpy(&root_data, lbvh.d_internal_data->data + root_id, sizeof(BBox3f), gpu_memcpy_device_to_host);

    printf(
        "Total bb : (%.5f, %.5f, %.5f) -- (%.5f, %.5f, %.5f)\n",
        root_data.min(0), root_data.min(1), root_data.min(2),
        root_data.max(0), root_data.max(1), root_data.max(2)
    );

    AvaDeviceArray<int, int>::Ptr d_ncoll = AvaDeviceArray<int, int>::create({n+1});
    AvaView<int, -1> d_ncoll_v = d_ncoll->to_view<-1>();
    uint32_t max_node_size[5] = {1, 2, 4, 8, 16};

    for (int test = 0; test < 5; test++){

        uint32_t LEAF_SIZE = max_node_size[test];

        gpu_device_synchronise();
        timespec_get(&t0, TIME_UTC);

        AvaView<Sphere3D, -1> d_obj_m_v = lbvh.d_obj_m->to_view<-1>();
        AvaView<uint32_t, -1> d_child_right_v = lbvh.d_child_right->to_view<-1>();
        AvaView<uint32_t, -1> d_child_left_v = lbvh.d_child_left->to_view<-1>();
        AvaView<uint32_t, -1> d_range_min_v = lbvh.d_range_min->to_view<-1>();
        AvaView<uint32_t, -1> d_range_max_v = lbvh.d_range_max->to_view<-1>();
        AvaView<uint32_t, -1> d_parent_v = lbvh.d_parent->to_view<-1>();
        AvaView<uint32_t, -1> d_root_v = lbvh.d_root->to_view<-1>();
        AvaView<BBox3f, -1> d_internal_data_v = lbvh.d_internal_data->to_view<-1>();
        uint32_t const n_v = n;

        ava_for<256>(nullptr, 0, n, [=] __device__ (int const tid) {
            // Stack for DFS on the tree
            uint32_t range_min;
            uint32_t range_max;
            uint32_t stack_size  = 0;
            uint32_t stack[64];
            stack[stack_size++] = d_root_v(0);

            int ncoll_loc = 0;
            Sphere3D const query = d_obj_m_v(tid);

            // DFS
            while (stack_size != 0) {
                uint32_t const cur = stack[--stack_size];
                uint32_t children[2] = {d_child_left_v(cur-n_v), d_child_right_v(cur-n_v)};

                for (int ichild = 0; ichild < 2; ichild++){
                    uint32_t const child_id = children[ichild];
                    if (child_id >= n_v) {
                        range_min = d_range_min_v(child_id-n_v);
                        range_max = d_range_max_v(child_id-n_v);
                    } else {
                        range_min = child_id;
                        range_max = child_id;
                    }
                    bool is_leaf = child_id < n_v || (range_max - range_min < LEAF_SIZE);

                    if (!is_leaf) {
                        BBox3f const node_data = d_internal_data_v(child_id);
                        bool is_in = true;
                        for (int i = 0; i < 3; i++){
                            is_in &= query.c[i] - query.r < node_data.max(i);
                            is_in &= query.c[i] + query.r > node_data.min(i);
                        }
                        if (is_in) stack[stack_size++] = child_id;
                    } else {  // The child is a leaf : compute
                        for (uint32_t obj_id = range_min; obj_id <= range_max; obj_id++){
                            const Sphere3D obj = d_obj_m_v(obj_id);
                            float const rad_sum = query.r + obj.r;
                            float const d2 = (query.c - obj.c).sqnorm();
                            if (d2 <= rad_sum*rad_sum && (tid != obj_id)) ncoll_loc++;
                        }
                    }
                }
            }

            d_ncoll_v(0) = 0;
            d_ncoll_v(tid+1) = ncoll_loc;
        });
        gpu_device_synchronise();
        timespec_get(&t1, TIME_UTC);
        printf("Collision count time with leaf size %u : %f ms\n", LEAF_SIZE,
                (t1.tv_sec - t0.tv_sec)*1e3 + (t1.tv_nsec - t0.tv_nsec)*1e-6);

        size_t tmp_size = 0;
        ava::scan::inplace_inclusive_sum(
            nullptr, 
            tmp_size, 
            d_ncoll->data+1,
            n
        );
        AvaDeviceArray<char, size_t>::Ptr tmp = AvaDeviceArray<char, size_t>::create({tmp_size});
        ava::scan::inplace_inclusive_sum(
            tmp->data, 
            tmp_size, 
            d_ncoll->data+1,
            n
        );

        int total;
        gpu_memcpy(&total, d_ncoll->data+n, sizeof(total), gpu_memcpy_device_to_host);
        printf("Total number of collisions [leaf size = %u] : %d\n", LEAF_SIZE, total);
    }


    // ================ Test with max depth ===========================

    gpu_device_synchronise();
    timespec_get(&t0, TIME_UTC);

    AvaView<Sphere3D, -1> d_obj_m_v = lbvh.d_obj_m->to_view<-1>();
    AvaView<uint32_t, -1> d_child_right_v = lbvh.d_child_right->to_view<-1>();
    AvaView<uint32_t, -1> d_child_left_v = lbvh.d_child_left->to_view<-1>();
    AvaView<uint32_t, -1> d_range_min_v = lbvh.d_range_min->to_view<-1>();
    AvaView<uint32_t, -1> d_range_max_v = lbvh.d_range_max->to_view<-1>();
    AvaView<uint32_t, -1> d_parent_v = lbvh.d_parent->to_view<-1>();
    AvaView<uint32_t, -1> d_root_v = lbvh.d_root->to_view<-1>();
    AvaView<BBox3f, -1> d_internal_data_v = lbvh.d_internal_data->to_view<-1>();
    uint32_t const n_v = n;

    ava_for<256>(nullptr, 0, n, [=] __device__ (int const tid) {
        // Stack for DFS on the tree
        uint32_t range_min;
        uint32_t range_max;
        uint32_t stack_size  = 0;
        uint32_t stack[32];
        stack[stack_size++] = d_root_v(0);

        int ncoll_loc = 0;
        Sphere3D const query = d_obj_m_v(tid);

        // DFS
        while (stack_size != 0) {
            uint32_t const cur = stack[--stack_size];
            uint32_t children[2] = {d_child_left_v(cur-n_v), d_child_right_v(cur-n_v)};

            for (int ichild = 0; ichild < 2; ichild++){
                uint32_t const child_id = children[ichild];
                bool is_leaf = (child_id < n_v) || (stack_size >= 32);

                if (!is_leaf) {
                    BBox3f const node_data = d_internal_data_v(child_id);
                    bool is_in = true;
                    for (int i = 0; i < 3; i++){
                        is_in &= query.c[i] - query.r < node_data.max(i);
                        is_in &= query.c[i] + query.r > node_data.min(i);
                    }
                    if (is_in) stack[stack_size++] = child_id;
                } else {  // The child is a leaf : compute
                    if (child_id >= n_v) {
                        range_min = d_range_min_v(child_id-n_v);
                        range_max = d_range_max_v(child_id-n_v);
                    } else {
                        range_min = child_id;
                        range_max = child_id;
                    }
                    for (uint32_t obj_id = range_min; obj_id <= range_max; obj_id++){
                        const Sphere3D obj = d_obj_m_v(obj_id);
                        float const rad_sum = query.r + obj.r;
                        float const d2 = (query.c - obj.c).sqnorm();
                        if (d2 <= rad_sum*rad_sum && (tid != obj_id)) ncoll_loc++;
                    }
                }
            }
        }

        d_ncoll_v(0) = 0;
        d_ncoll_v(tid+1) = ncoll_loc;
    });
    gpu_device_synchronise();
    timespec_get(&t1, TIME_UTC);
    printf("Collision count time with max depth 32 : %f ms\n", 
            (t1.tv_sec - t0.tv_sec)*1e3 + (t1.tv_nsec - t0.tv_nsec)*1e-6);

    size_t tmp_size = 0;
    ava::scan::inplace_inclusive_sum(
        nullptr, 
        tmp_size, 
        d_ncoll->data+1,
        n
    );
    AvaDeviceArray<char, size_t>::Ptr tmp = AvaDeviceArray<char, size_t>::create({tmp_size});
    ava::scan::inplace_inclusive_sum(
        tmp->data, 
        tmp_size, 
        d_ncoll->data+1,
        n
    );

    int total;
    gpu_memcpy(&total, d_ncoll->data+n, sizeof(total), gpu_memcpy_device_to_host);
    printf("Total number of collisions [max depth = 32] : %d\n", total);
    return EXIT_SUCCESS;
}
