#include <cstdlib>
#include <ctime>
#include <vector>
#include "ava.h"
#include "ava_device_array.h"
#include "ava_host_array.h"
#include "ava_host_array.hpp"
#include "bbox.hpp"
#include "vec.hpp"

#include "mesh/mesh.hpp"

int main(int argc, char** argv) {

    struct timespec t0, t1;

    uint32_t n = 100;

    if (argc > 1) {
        n = strtoul(argv[1], NULL, 10);
    }

    std::vector<Vec2f> h_coords;
    for (uint32_t i = 0; i < n; i++){
        Vec2f tmp = {
            (fp_tt) rand() / RAND_MAX,
            (fp_tt) rand() / RAND_MAX
        };
        h_coords.emplace_back(tmp);
    }

    AvaDeviceArray<Vec2f, int>::Ptr d_coords = AvaDeviceArray<Vec2f, int>::create({(int) h_coords.size()});
    deep_copy(d_coords->data, h_coords.data(), h_coords.size());

    Mesh2D mesh;
    mesh.d_nodes = d_coords;

    printf(
        "========== Mesh2D Data Type sizes ===========\n"
        "\t Prim : %zu bytes\n"
        "\t Elem : %zu bytes\n"
        "\t LocalElem : %zu bytes\n"
        "\t VecT : %zu bytes\n"
        "\t BBoxT : %zu bytes\n",
        sizeof(Mesh2D::Prim), 
        sizeof(Mesh2D::Elem),
        sizeof(Mesh2D::LocalElem),
        sizeof(Mesh2D::VecT),
        sizeof(Mesh2D::BBoxT)
    );

    timespec_get(&t0, TIME_UTC);
    mesh.init();
    gpu_device_synchronise();
    timespec_get(&t1, TIME_UTC);
    printf("Init time : %.5f\n", (t1.tv_sec - t0.tv_sec)*1e3 + (t1.tv_nsec - t0.tv_nsec)*1e-6);

    timespec_get(&t0, TIME_UTC);
    mesh.insert_morton_neighbors();
    gpu_device_synchronise();
    timespec_get(&t1, TIME_UTC);
    printf("Morton insert time : %.5f\n", (t1.tv_sec - t0.tv_sec)*1e3 + (t1.tv_nsec - t0.tv_nsec)*1e-6);

    timespec_get(&t0, TIME_UTC);
    mesh.insert_by_circumsphere_checking();
    gpu_device_synchronise();
    timespec_get(&t1, TIME_UTC);
    printf("Circumsphere insertion time : %.5f\n", (t1.tv_sec - t0.tv_sec)*1e3 + (t1.tv_nsec - t0.tv_nsec)*1e-6);

    uint32_t root_id = 0;
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

    AvaHostArray<uint8_t, int>::Ptr h_node_is_complete = AvaHostArray<uint8_t, int>::create({(int) mesh.n_nodes});
    gpu_memcpy(h_node_is_complete->data(), mesh.d_node_is_complete->data, sizeof(uint8_t)*h_node_is_complete->size(), gpu_memcpy_device_to_host);
    FILE* fnode_complete = fopen("node_is_complete.txt", "w+");
    for (uint32_t i = 0; i < h_node_is_complete->size(); i++){
        fprintf(fnode_complete, "%u\n", h_node_is_complete(i));
    }
    fclose(fnode_complete);

    return 0;
}
