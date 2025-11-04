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
#endif

    AvaHostArray<Vec2f>::Ptr h_coords({
        {0.0f, 0.0f},
        {1.0f, 0.0f},
        {1.0f, 1.0f},
        {0.0f, 1.0f},
        {0.25f, 0.25f},
        {0.75f, 0.25f},
        {0.75f, 0.75f},
        {0.25f, 0.75f},
        {0.5f, 0.5f},
        {0.4f, 0.4f},
        {0.6f, 0.6f},
        {0.1f, 0.6f}
    });

    AvaDeviceArray<Vec2f, int>::Ptr d_coords = AvaDeviceArray<Vec2f, int>::create({h_coords->size()});
    deep_copy(d_coords->data, h_coords->data(), h_coords->size());

    Mesh2D mesh;
    mesh.d_nodes = d_coords;

    mesh.init();
    mesh.insert_morton_neighbors();
    mesh.insert_by_circumsphere_checking();

    uint32_t root_id;
    gpu_memcpy(&root_id, mesh.lbvh.d_root->data, sizeof(root_id), gpu_memcpy_device_to_host);

    BBox2f root_data;
    gpu_memcpy(&root_data, mesh.lbvh.d_internal_data->data + root_id, sizeof(root_data), gpu_memcpy_device_to_host);
    printf(
        "===================== Bounding box ==================\n"
        "Total bb : (%.5f, %.5f) -- (%.5f, %.5f)\n",
        root_data.min(0), root_data.min(1),
        root_data.max(0), root_data.max(1)
    );


    FILE* fnode = fopen("nodes.txt", "w+");
    AvaHostArray<Vec2f, int>::Ptr h_nodes_m = AvaHostArray<Vec2f, int>::create({mesh.lbvh.d_obj_m->size});
    gpu_memcpy(h_nodes_m->data(), mesh.lbvh.d_obj_m->data, sizeof(Vec2f)*h_nodes_m->size(), gpu_memcpy_device_to_host);
    for (uint32_t i = 0; i < h_nodes_m->size(); i++) {
        fprintf(fnode, "%.5f %.5f\n", h_nodes_m(i)[0], h_nodes_m(i)[1]);
    }
    fclose(fnode);

    mesh.compress_into_global();

    AvaHostArray<Mesh2D::Elem, int>::Ptr h_elem = AvaHostArray<Mesh2D::Elem, int>::create({mesh.d_elemglob->size});
    gpu_memcpy(h_elem->data(), mesh.d_elemglob->data, sizeof(Mesh2D::Elem)*h_elem->size(), gpu_memcpy_device_to_host);
    FILE* ftri = fopen("elem.txt", "w+");
    for (uint32_t i = 0; i < h_elem->size(); i++){
        Mesh2D::Elem elem = h_elem(i);
        fprintf(ftri, "%u %u %u\n", elem.a, elem.b, elem.c);
    }
    fclose(ftri);

    return 0;
}
