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
#include "ava.h"
#include "ava_device_array.h"
#include "ava_host_array.h"
#include "ava_host_array.hpp"
#include "bbox.hpp"
#include "vec.hpp"

#include "mesh/mesh.hpp"

int main(void) {

#ifndef AVA_TARGET_CPU 
    printf("This is a debug test that can only be performed on a CPU architecture\n");
    return 0;
#else

    AvaHostArray<Vec2f>::Ptr h_coords({
        {0.0f, 0.0f},
        {1.0f, 0.0f},
        {1.0f, 1.0f},
        {0.0f, 1.0f},
    });

    AvaDeviceArray<Vec2f, int>::Ptr d_coords = AvaDeviceArray<Vec2f, int>::create({h_coords->size()});
    deep_copy(d_coords->data, h_coords->data(), h_coords->size());

    Mesh2D mesh;
    mesh.d_nodes = d_coords;

    mesh.init();

    uint32_t root_id = mesh.lbvh.d_root->data[0];
    BBox2f root_data = mesh.lbvh.d_internal_data->data[root_id];
    printf(
        "===================== Bounding box ==================\n"
        "Total bb : (%.5f, %.5f) -- (%.5f, %.5f)\n",
        root_data.min(0), root_data.min(1),
        root_data.max(0), root_data.max(1)
    );

    printf("==================== All nodes ======================\n");
    for (int i = 0; i < mesh.lbvh.d_obj_m->size; i++) {
        printf("\tNode %u : (%.5f, %.5f)\n", 
                i,
                mesh.lbvh.d_obj_m->data[i][0],
                mesh.lbvh.d_obj_m->data[i][1]
            );
    }

    printf("=================== Local Triangulation (per node) ================\n");
    Mesh2D::TriLoc tloc_v = mesh.get_triloc_struct();
    for (uint32_t i = 0; i < mesh.n_nodes; i++) {
        Mesh2D::TriLoc tloc = tloc_v.thread_init(i);
        printf("tid = %u :\n", i);
        for (uint32_t j = 0; j < mesh.d_node_nelemloc->data[i]; j++){
            Mesh2D::LocalElem const elem_loc = tloc.get_elem(j);
            printf("\t%u-th elem : local indices : (%u, %u) <==> global indices : (%u, %u)\n", 
                    j, 
                    elem_loc.a, elem_loc.b, 
                    tloc.get_neig(elem_loc.a), tloc.get_neig(elem_loc.b)
                  );
        }
    }

    printf("=================== Local Triangulations (memory layout) =================\n");
    for (int i = 0; i < mesh.d_elemloc->size; i++){
        printf("\tblock size : (%u, %u), block col %u. tri %u in local indices (%u, %u)\n", 
                mesh.cur_max_nelem, 32,
                i % 32,
                (i/32) % 32,
                mesh.d_elemloc->data[i].a,
                mesh.d_elemloc->data[i].b
            );
    }

    return 0;
#endif
}
