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
 * Test LBVH from [Karras, 2012] on random spheres in 3D
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
#include "lbvh.hpp"
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
    Sphere3DLBVH lbvh;
    lbvh.set_objects(d_coords);
    struct timespec t0, t1;
    ava_device_sync();
    timespec_get(&t0, TIME_UTC);
    lbvh.build();
    ava_device_sync();
    timespec_get(&t1, TIME_UTC);
    printf("Construction time : %f ms\n", 
            (t1.tv_sec - t0.tv_sec)*1e3 + (t1.tv_nsec - t0.tv_nsec)*1e-6);

    BBox3f root_data;
    deep_copy(&root_data, lbvh.d_internal_data->data, 1);

    printf(
        "Total bb : (%.5f, %.5f, %.5f) -- (%.5f, %.5f, %.5f)\n",
        root_data.min(0), root_data.min(1), root_data.min(2),
        root_data.max(0), root_data.max(1), root_data.max(2)
    );

    AvaDeviceArray<int, int>::Ptr d_ncoll = AvaDeviceArray<int, int>::create({n+1});
    AvaView<int, -1> d_ncoll_v = d_ncoll->to_view<-1>();

    ava_device_sync();
    timespec_get(&t0, TIME_UTC);

    AvaView<Sphere3D, -1> d_obj_m_v = lbvh.d_obj_m->to_view<-1>();
    AvaView<int, -1> d_internal_sep_v = lbvh.d_internal_sep->to_view<-1>();
    AvaView<uint8_t, -1> d_child_is_leaf_v = lbvh.d_child_is_leaf->to_view<-1>();
    AvaView<BBox3f, -1> d_internal_data_v = lbvh.d_internal_data->to_view<-1>();

    ava_for<256>(nullptr, 0, n, [=] __device__ (int const tid) {
        // Stack for DFS on the tree
        int stack_size  = 0;
        int stack[64];
        stack[stack_size++] = 0;

        int ncoll_loc = 0;
        Sphere3D const query = d_obj_m_v(tid);

        // DFS
        while (stack_size != 0) {
            int const cur = stack[--stack_size];
            int const internal_sep = d_internal_sep_v(cur); 
            uint8_t const child_is_leaf = d_child_is_leaf_v(cur);

            for (int ichild = 0; ichild < 2; ichild++){
                // Child is an internal node
                if (!(child_is_leaf & (ichild+1))) { 
                    BBox3f const node_data = d_internal_data_v(internal_sep+ichild);
                    bool is_in = true;
                    for (int i = 0; i < 3; i++){
                        is_in &= query.c[i] - query.r < node_data.max(i);
                        is_in &= query.c[i] + query.r > node_data.min(i);
                    }
                    if (is_in) stack[stack_size++] = internal_sep+ichild;

                } else {  // The child is a leaf : compute
                    const Sphere3D obj = d_obj_m_v(internal_sep+ichild);
                    float const rad_sum = query.r + obj.r;
                    float const d2 = (query.c - obj.c).sqnorm();
                    if (d2 <= rad_sum*rad_sum && (tid != internal_sep+ichild)) ncoll_loc++;
                }
            }
        }

        d_ncoll_v(0) = 0;
        d_ncoll_v(tid+1) = ncoll_loc;
    });
    ava_device_sync();
    timespec_get(&t1, TIME_UTC);
    printf("Collision count time : %f ms\n", 
            (t1.tv_sec - t0.tv_sec)*1e3 + (t1.tv_nsec - t0.tv_nsec)*1e-6);

    size_t tmp_size = 0;
    ERRCHK(ava::scan::inplace_inclusive_sum(
        nullptr, 
        tmp_size, 
        d_ncoll->data+1,
        n
    ));
    AvaDeviceArray<char, size_t>::Ptr tmp = AvaDeviceArray<char, size_t>::create({tmp_size});
    ERRCHK(ava::scan::inplace_inclusive_sum(
        tmp->data, 
        tmp_size, 
        d_ncoll->data+1,
        n
    ));

    int total;
    deep_copy(&total, d_ncoll->data+n, 1);
    printf("Total number of collisions : %d\n", total);

    return EXIT_SUCCESS;
}
