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
/*
 * Test LBVH from [Apetrei, 2014] on a set of sphere in 2D on a line. 
 * The number of collisions is known in advance
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
    AvaHostArray<Sphere2D>::Ptr h_coords = AvaHostArray<Sphere2D>::create({n});
    for (int i = 0; i < n; i++){
        Sphere2D s;
        s.c[0] = i;
        s.c[1] = 0.0f;
        s.r = 0.55f;
        h_coords(i) = s;
    }

    AvaDeviceArray<Sphere2D, int>::Ptr d_coords = AvaDeviceArray<Sphere2D, int>::create({n});
    d_coords->set(h_coords);

    // Construct lbvh
    Sphere2DLBVHa lbvh;
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
    deep_copy(&root_id, lbvh.d_root->data, 1);

    BBox2f root_data;
    deep_copy(&root_data, lbvh.d_internal_data->data + root_id, 1);

    printf(
        "Total bb : (%.5f, %.5f) -- (%.5f, %.5f)\n",
        root_data.min(0), root_data.min(1),
        root_data.max(0), root_data.max(1)
    );

    if (root_data.min(0) != -0.55f || root_data.max(0) != ((n-1)*1.0f + 0.55f) 
            || root_data.min(1) != -0.55f || root_data.max(1) != 0.55f) {
        printf("Global bounding box is incorrect, expected : [-0.55, 100.55] x [-0.55, 0.55]\n");
    }


    AvaDeviceArray<int, int>::Ptr d_ncoll = AvaDeviceArray<int, int>::create({n+1});
    AvaView<int, -1> d_ncoll_v = d_ncoll->to_view<-1>();
    uint32_t max_node_size[5] = {1, 2, 4, 8, 16};

    for (int test = 0; test < 5; test++){

        uint32_t LEAF_SIZE = max_node_size[test];

        gpu_device_synchronise();
        timespec_get(&t0, TIME_UTC);

        AvaView<Sphere2D, -1> d_obj_m_v = lbvh.d_obj_m->to_view<-1>();
        AvaView<uint32_t, -1, 2> d_children_v = lbvh.d_children->to_view<-1, 2>();
        AvaView<uint32_t, -1, 2> d_range_v = lbvh.d_range->to_view<-1, 2>();
        AvaView<uint32_t, -1> d_root_v = lbvh.d_root->to_view<-1>();
        AvaView<BBox2f, -1> d_internal_data_v = lbvh.d_internal_data->to_view<-1>();
        uint32_t const n_v = n;

        ava_for<256>(nullptr, 0, n, [=] __device__ (uint32_t const tid) {
            // Stack for DFS on the tree
            uint32_t range_min;
            uint32_t range_max;
            uint32_t stack_size  = 0;
            uint32_t stack[64];
            stack[stack_size++] = d_root_v(0);

            uint32_t ncoll_loc = 0;
            Sphere2D const query = d_obj_m_v(tid);

            // DFS
            while (stack_size != 0) {
                uint32_t const cur = stack[--stack_size];

                for (int ichild = 0; ichild < 2; ichild++){
                    uint32_t const child_id = d_children_v(cur-n_v, ichild);
                    if (child_id >= n_v) {
                        range_min = d_range_v(child_id-n_v, 0);
                        range_max = d_range_v(child_id-n_v, 1);
                    } else {
                        range_min = child_id;
                        range_max = child_id;
                    }
                    bool is_leaf = child_id < n_v || (range_max - range_min < LEAF_SIZE);

                    if (!is_leaf) {
                        BBox2f const node_data = d_internal_data_v(child_id);
                        bool is_in = true;
                        for (int i = 0; i < 2; i++){
                            is_in &= query.c[i] - query.r < node_data.max(i);
                            is_in &= query.c[i] + query.r > node_data.min(i);
                        }
                        if (is_in) stack[stack_size++] = child_id;
                    } else {  // The child is a leaf : compute
                        for (uint32_t obj_id = range_min; obj_id <= range_max; obj_id++){
                            const Sphere2D obj = d_obj_m_v(obj_id);
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
        deep_copy(&total, d_ncoll->data+n, 1);
        printf("Total number of collisions [leaf size = %u] : %d\n", LEAF_SIZE, total);

        if (total != (n-2)*2 + 2) {
            printf("Incorrect total number of collisions, expected %d\n", (n-2)*2 + 2);
        }
    }


    // ================ Test with max depth ===========================

    gpu_device_synchronise();
    timespec_get(&t0, TIME_UTC);

    AvaView<Sphere2D, -1> d_obj_m_v = lbvh.d_obj_m->to_view<-1>();
    AvaView<uint32_t, -1, 2> d_children_v = lbvh.d_children->to_view<-1, 2>();
    AvaView<uint32_t, -1, 2> d_range_v = lbvh.d_range->to_view<-1, 2>();
    AvaView<uint32_t, -1> d_root_v = lbvh.d_root->to_view<-1>();
    AvaView<BBox2f, -1> d_internal_data_v = lbvh.d_internal_data->to_view<-1>();
    uint32_t const n_v = n;

    ava_for<256>(nullptr, 0, n, [=] __device__ (uint32_t const tid) {
        // Stack for DFS on the tree
        uint32_t range_min;
        uint32_t range_max;
        uint32_t stack_size  = 0;
        uint32_t stack[32];
        stack[stack_size++] = d_root_v(0);

        uint32_t ncoll_loc = 0;
        Sphere2D const query = d_obj_m_v(tid);

        // DFS
        while (stack_size != 0) {
            uint32_t const cur = stack[--stack_size];

            for (int ichild = 0; ichild < 2; ichild++){
                uint32_t const child_id = d_children_v(cur-n_v, ichild);
                bool is_leaf = (child_id < n_v) || (stack_size >= 32);

                if (!is_leaf) {
                    BBox2f const node_data = d_internal_data_v(child_id);
                    bool is_in = true;
                    for (int i = 0; i < 2; i++){
                        is_in &= query.c[i] - query.r < node_data.max(i);
                        is_in &= query.c[i] + query.r > node_data.min(i);
                    }
                    if (is_in) stack[stack_size++] = child_id;
                } else {  // The child is a leaf : compute
                    if (child_id >= n_v) {
                        range_min = d_range_v(child_id-n_v, 0);
                        range_max = d_range_v(child_id-n_v, 1);
                    } else {
                        range_min = child_id;
                        range_max = child_id;
                    }
                    for (uint32_t obj_id = range_min; obj_id <= range_max; obj_id++){
                        const Sphere2D obj = d_obj_m_v(obj_id);
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
    printf("Collision count time with max depth 16 : %f ms\n", 
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
    deep_copy(&total, d_ncoll->data+n, 1);
    printf("Total number of collisions [max depth = 32] : %d\n", total);

    if (total != (n-2)*2 + 2) {
        printf("Incorrect total number of collisions, expected %d\n", (n-2)*2 + 2);
    }
    return EXIT_SUCCESS;
}
