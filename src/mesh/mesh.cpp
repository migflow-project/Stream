#include <cfloat>
#include <cmath>
#include <cstdio>
#include <ctime>
#include "ava_device_array.h"
#include "ava_host_array.h"
#include "ava_host_array.hpp"
#include "ava_view.h"
#include "ava_scan.h"
#include "ava_select.h"
#include "defines.h"
#include "predicates.hpp"
#include "traversal_stack.hpp"
#include "vec.hpp"
#include "mesh.hpp"

template struct stream::mesh::Mesh<2>;

namespace stream::mesh {

    using Heap = geo::TraversalMinHeap<uint32_t, fp_tt, 32>;
    using Stack = geo::TraversalStack<uint32_t, 32>;

    template<int dim, uint32_t block_size>
    Mesh<dim, block_size>::Mesh() noexcept {
        n_nodes = 0;
        n_edges = 0;
        n_elems = 0;
        n_blocks = 0;
        cur_max_nneig = n_neig_guess;
        cur_max_nelem = n_init_local_elem;

        temp_mem_size = 0;
        d_temp_mem = AvaDeviceArray<uint8_t, size_t>::create({0});
                           
        d_elemloc = AvaDeviceArray<LocalElem, int>::create({0});
        d_node_is_complete = AvaDeviceArray<uint8_t, int>::create({0});
        d_node_nelemloc = AvaDeviceArray<uint32_t, int>::create({0});
        d_node_nneigloc = AvaDeviceArray<uint32_t, int>::create({0});
        d_node_nelem_out = AvaDeviceArray<uint32_t, int>::create({0});
        d_elemglob = AvaDeviceArray<Elem, int>::create({0});
        d_neig = AvaDeviceArray<uint32_t, int>::create({0});
    };


    template<int dim, uint32_t block_size>
    void Mesh<dim, block_size>::init() {

        n_nodes = d_nodes->size;
        n_edges = 0;
        n_elems = 0;
        n_blocks = (n_nodes + block_size - 1)/block_size;
        cur_max_nneig = n_neig_guess;
        cur_max_nelem = n_init_local_elem;

        d_elemloc->resize({(int) (n_blocks * block_size * n_init_local_elem)});
        d_node_is_complete->resize({(int) n_nodes});
        d_node_nelemloc->resize({(int) n_nodes});
        d_node_nneigloc->resize({(int) n_nodes});
        d_node_nelem_out->resize({(int) (n_nodes+1)});
        d_neig->resize({(int) (n_blocks * block_size * cur_max_nneig)});

        lbvh.set_objects(d_nodes);
        lbvh.build();

        // Set infinity points at the end of the morton-reordered nodes
        uint32_t n_points_v = n_nodes;
        lbvh.d_obj_m->resize({(int) (n_nodes+n_inf_nodes)});

        AvaView<VecT, -1> d_coords_m_v = lbvh.d_obj_m->template to_view<-1>();
        AvaView<uint32_t, -1> d_node_nelemloc_v = d_node_nelemloc->to_view<-1>();
        AvaView<uint32_t, -1> d_node_nneigloc_v = d_node_nneigloc->to_view<-1>();
        BBoxT const * const __restrict__ d_bb_glob = lbvh.d_bb_glob->data;
        TriLoc tloc = get_triloc_struct();
        ava_for<256>(nullptr, 0, n_nodes, [=] __device__ (int const tid) {

            // Only first thread initialize the infinity nodes
            if (tid == 0) {
                VecT const avg = 0.5f*(d_bb_glob->pmax + d_bb_glob->pmin);
                VecT const range = d_bb_glob->pmax - d_bb_glob->pmin;
                d_coords_m_v;
                n_points_v;

                if constexpr (dim == 2) {
                    fp_tt const max_range = std::fmax(range[0], range[1]);

                    // Bottom left
                    d_coords_m_v(n_points_v) = {avg[0] - 2*max_range, avg[1] - 2*max_range};

                    // Bottom right
                    d_coords_m_v(n_points_v+1) = {avg[0] + 2*max_range, avg[1] - 2*max_range};

                    // Top
                    d_coords_m_v(n_points_v+2) = {avg[0], avg[1] + 2*max_range};

                } else {
                    fp_tt const max_range = std::fmax(range[0], std::fmax(range[1], range[2]));

                    // First vertex of base
                    d_coords_m_v(n_points_v) = {avg[0] - 2*max_range, avg[1] - 2*max_range, avg[2] - 2*max_range};

                    // Second vertex of base
                    d_coords_m_v(n_points_v + 1) = {avg[0] + 2*max_range, avg[1] - 2*max_range, avg[2] - 2*max_range};

                    // Third vertex of base
                    d_coords_m_v(n_points_v + 2) = {avg[0] + 2*max_range, avg[1] + 2*max_range, avg[2] - 2*max_range};

                    // Top vertex
                    d_coords_m_v(n_points_v + 3) = {avg[0], avg[1], avg[2] + 2*max_range};
                }
            }

            // Initialize local triangulation
            uint32_t const elem_start   = tloc.get_elem_offset(tid); 
            d_node_nelemloc_v(tid) = n_init_elem;
            d_node_nneigloc_v(tid) = n_inf_nodes;
            //  ================ Init cell ===================

            if constexpr (dim==2) {
                // Top left triangle
                tloc.get_elem(elem_start, 0) = {1, 2};

                // Top right
                tloc.get_elem(elem_start, 1) = {2, 0};

                // Bottom right
                tloc.get_elem(elem_start, 2) = {0, 1};
            } else {
                // bottom tet
                tloc.get_elem(elem_start, 0) = {0, 2, 1};

                // tet facing you
                tloc.get_elem(elem_start, 1) = {1, 2, 3};

                // left tet
                tloc.get_elem(elem_start, 2) = {0, 1, 3};

                // right tet
                tloc.get_elem(elem_start, 3) = {0, 3, 2};
            }
        });
    }

    template<int dim, uint32_t block_size>
    void Mesh<dim, block_size>::insert_morton_neighbors() {

        uint32_t const n_nodes_v = n_nodes;

        TriLoc tloc = get_triloc_struct();
        AvaView<VecT, -1> d_nodes_m_v = lbvh.d_obj_m->template to_view<-1>();
        AvaView<uint32_t, -1> d_node_nelemloc_v = d_node_nelemloc->to_view<-1>();
        AvaView<uint32_t, -1> d_node_nneigloc_v = d_node_nneigloc->to_view<-1>();

        ava_for<256>(nullptr, 0, n_nodes_v, [=] __device__ (uint32_t const tid) {

            // Get offsets of neighbors and triangles in the blocks
            uint32_t const neig_offset = tloc.get_neig_offset(tid); 
            uint32_t const elem_offset  = tloc.get_elem_offset(tid); 

            // Get the range of morton neighbors to insert. Take care of 
            // edge cases to ensure all nodes insert the same amount.
            uint32_t insert_start_range = tid - n_init_insert/2;
            uint32_t insert_end_range = tid + n_init_insert/2;
            if (tid < n_init_insert/2) {
                insert_start_range = 0;
                insert_end_range = n_nodes_v < n_init_insert ? n_nodes_v : n_init_insert + 1;
            } 
            if (tid + n_init_insert/2 >= n_nodes_v) {
                insert_start_range = n_nodes_v < n_init_insert ? 0 : n_nodes_v - n_init_insert;
                insert_end_range = n_nodes_v;
            }

            uint32_t ti;
            uint32_t Tlast = d_node_nelemloc_v(tid);
            uint32_t cur_neig_loc = d_node_nneigloc_v(tid);

            for (uint32_t ni = 0; ni < insert_end_range-insert_start_range; ni++) {

                // Get local/global indices of neighbor
                uint32_t cur_neig_glob = insert_start_range + ni;
                if (cur_neig_glob == tid) continue; // Do not insert the node itself

                tloc.set_neig(neig_offset, cur_neig_loc) = cur_neig_glob;

                // 256-bits bitset to indicate if i-th neighbor is in the cavity
                // of the inserted point
                uint64_t neig_in_cavity[4] = {0, 0, 0, 0};

                // Look at every triangle in the triangulation and add it to the 
                // cavity if the inserted point is inside its circumcircle
                uint32_t cavity_size = 0;
                ti = 0;
                while (ti < Tlast) {
                    // Get local triangle made by tid and two neighbors
                    LocalElem const elem_loc = tloc.get_elem(elem_offset, ti);

                    uint32_t i1 = tid;
                    uint32_t i2 = tloc.get_neig(neig_offset, elem_loc.a);
                    uint32_t i3 = tloc.get_neig(neig_offset, elem_loc.b);
                    uint32_t i4 = cur_neig_glob;
                    
                    fp_tt const det = incircle_SoS(i1, i2, i3, i4, d_nodes_m_v);

                    // If the inserted point is in the circumcircle of the triangle, 
                    // add the triangle's edges to the cavity and remove it from the local 
                    // triangulation
                    if (det > 0.0f) {
                        // Add T[ti] to the used edges
                        neig_in_cavity[elem_loc.a >> 6] ^= 1ULL << (elem_loc.a & 63);
                        neig_in_cavity[elem_loc.b >> 6] ^= 1ULL << (elem_loc.b & 63);
                        cavity_size++;

                        // Remove T[ti] from T
                        Tlast--;
                        tloc.get_elem(elem_offset, ti) = tloc.get_elem(elem_offset, Tlast);
                    } else {
                        ti++;
                    }
                }

                if (cavity_size) {
                    // If the cavity is not empty, retriangulate the cavity
                    // Special case for 2D : there are only 2 elements in neig_in_cavity
                    for (uint8_t r = 0; r < cur_neig_loc; r++){
                        if (neig_in_cavity[r >> 6] & (1ULL << (r & 63))) {
                            tloc.get_elem(elem_offset, Tlast) = {(uint8_t) r, (uint8_t) cur_neig_loc};
                            Tlast++;
                        }
                    }
                    cur_neig_loc++;
                } 
            }

            d_node_nelemloc_v(tid) = Tlast;
            d_node_nneigloc_v(tid) = cur_neig_loc;
        });
    }

    template<int dim, uint32_t block_size>
    void Mesh<dim, block_size>::insert_quadrant_neighbors() {

        uint32_t const n_nodes_v = n_nodes;

        TriLoc tloc = get_triloc_struct();
        AvaView<VecT, -1> d_nodes_m_v = lbvh.d_obj_m->template to_view<-1>();
        AvaView<uint32_t, -1> d_node_nelemloc_v = d_node_nelemloc->to_view<-1>();
        AvaView<uint32_t, -1> d_node_nneigloc_v = d_node_nneigloc->to_view<-1>();

        ava_for<256>(nullptr, 0, n_nodes_v, [=] __device__ (uint32_t const tid) {

            // Get offsets of neighbors and triangles in the blocks
            uint32_t const neig_offset = tloc.get_neig_offset(tid); 
            uint32_t const elem_offset  = tloc.get_elem_offset(tid); 

            // Find one point in each quadrant around the node
            VecT const p1 = d_nodes_m_v(tid);
            uint32_t to_insert[8] = {
                tid, tid, tid, tid,
                tid, tid, tid, tid
            };
            fp_tt dist[8] = {
                FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX,
                FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX
            };
            uint32_t const start = (tid > 32) ? tid - 32 : 0;
            uint32_t const end = (tid + 32 < n_nodes_v) ? tid + 32 : n_nodes_v;
            for (uint32_t i = start; i < end; i++) {
                if (i == tid) continue;
                VecT const p2  = d_nodes_m_v(i);
                fp_tt const d2 = (p1 - p2).sqnorm();

                // Get the quadrant in which p2 is located w.r.t p1
                uint32_t const id = ((p2[0] > p1[0]) << 1) | (p2[1] > p1[1]);
                if (d2 < dist[2*id]) {
                    to_insert[2*id] = i;
                    dist[2*id] = d2;

                    if (d2 < dist[2*id+1]) {
                        uint32_t tmp = to_insert[2*id+1];
                        to_insert[2*id+1] = i;
                        to_insert[2*id] = tmp;

                        fp_tt tmp_f = dist[2*id+1];
                        dist[2*id+1] = d2;
                        dist[2*id] = tmp_f;
                    }  
                }
            }


            uint32_t Tlast = d_node_nelemloc_v(tid);
            uint32_t cur_neig_loc = d_node_nneigloc_v(tid);

            for (uint32_t ni = 0; ni < 8; ni++) {

                // Get local/global indices of neighbor
                uint32_t cur_neig_glob = to_insert[ni];
                if (cur_neig_glob == tid) continue; // Do not insert the node itself

                tloc.set_neig(neig_offset, cur_neig_loc) = cur_neig_glob;


                // 256-bits bitset to indicate if i-th neighbor is in the cavity
                // of the inserted point
                uint64_t neig_in_cavity[4] = {0, 0, 0, 0};

                // Look at every triangle in the triangulation and add it to the 
                // cavity if the inserted point is inside its circumcircle
                uint32_t cavity_size = 0;
                uint32_t ti = 0;
                while (ti < Tlast) {
                    // Get local triangle made by tid and two neighbors
                    LocalElem const elem_loc = tloc.get_elem(elem_offset, ti);

                    uint32_t i1 = tid;
                    uint32_t i2 = tloc.get_neig(neig_offset, elem_loc.a);
                    uint32_t i3 = tloc.get_neig(neig_offset, elem_loc.b);
                    uint32_t i4 = cur_neig_glob;
                    
                    fp_tt const det = incircle_SoS(i1, i2, i3, i4, d_nodes_m_v);

                    // If the inserted point is in the circumcircle of the triangle, 
                    // add the triangle's edges to the cavity and remove it from the local 
                    // triangulation
                    if (det > 0.0f) {
                        // Add T[ti] to the used edges
                        neig_in_cavity[elem_loc.a >> 6] ^= 1ULL << (elem_loc.a & 63);
                        neig_in_cavity[elem_loc.b >> 6] ^= 1ULL << (elem_loc.b & 63);
                        cavity_size++;

                        // Remove T[ti] from T
                        Tlast--;
                        tloc.get_elem(elem_offset, ti) = tloc.get_elem(elem_offset, Tlast);
                    } else {
                        ti++;
                    }
                }

                if (cavity_size) {
                    // If the cavity is not empty, retriangulate the cavity
                    // Special case for 2D : there are only 2 elements in neig_in_cavity
                    for (uint8_t r = 0; r < cur_neig_loc; r++){
                        if (neig_in_cavity[r >> 6] & (1ULL << (r & 63))) {
                            tloc.get_elem(elem_offset, Tlast) = {(uint8_t) r, (uint8_t) cur_neig_loc};
                            Tlast++;
                        }
                    }
                    cur_neig_loc++;
                } 
            }

            d_node_nelemloc_v(tid) = Tlast;
            d_node_nneigloc_v(tid) = cur_neig_loc;
        });
    }

    template<int dim, uint32_t block_size>
    void Mesh<dim, block_size>::insert_BVH_neighbors() {
        uint32_t const n_nodes_v = n_nodes;

        TriLoc tloc = get_triloc_struct();
        AvaView<VecT, -1> d_nodes_m_v = lbvh.d_obj_m->template to_view<-1>();
        AvaView<uint32_t, -1> d_node_nelemloc_v = d_node_nelemloc->to_view<-1>();
        AvaView<uint32_t, -1> d_node_nneigloc_v = d_node_nneigloc->to_view<-1>();

        AvaView<uint32_t, -1> d_root_v = lbvh.d_root->template to_view<-1>();
        AvaView<uint32_t, -1> d_split_idx_v = lbvh.d_split_idx->template to_view<-1>();
        AvaView<uint32_t, -1> d_child_left_v = lbvh.d_child_left->template to_view<-1>();
        AvaView<uint32_t, -1> d_child_right_v = lbvh.d_child_right->template to_view<-1>();
        AvaView<uint32_t, -1> d_range_max_v = lbvh.d_range_max->template to_view<-1>();
        AvaView<uint32_t, -1> d_range_min_v = lbvh.d_range_min->template to_view<-1>();
        AvaView<uint64_t, -1> d_morton_v = lbvh.d_morton_sorted->template to_view<-1>();
        AvaView<BBoxT, -1>    d_bb_glob_v = lbvh.d_bb_glob->template to_view<-1>();
        AvaView<BBoxT, -1>    d_bboxes_v = lbvh.d_internal_data->template to_view<-1>();

        typename AvaDeviceArray<BBoxT, int>::Ptr d_boxes = AvaDeviceArray<BBoxT, int>::create({(int) lbvh.d_internal_data->size});
        AvaView<BBoxT, -1> d_boxes_v = d_boxes->template to_view<-1>();
        ava_for<256>(nullptr, 0, n_nodes_v, [=] __device__ (uint32_t const tid) {
            // Fit the bboxes of the nodes
            BBoxT bb = d_bb_glob_v(0);
            uint32_t range_min;
            uint32_t range_max;

            uint32_t cur = d_root_v(0);
            bool leaf_reached = false;
            while (!leaf_reached) {
                // Get the splitting plane of this internal node
                uint32_t const split_idx = d_split_idx_v(cur-n_nodes_v);
                uint32_t const split_bit_idx = __builtin_clzll(d_morton_v(split_idx) ^ d_morton_v(split_idx+1));
                uint32_t const split_axis = (split_bit_idx+(dim-1)) % dim;

                uint32_t const children[2] = {d_child_left_v(cur-n_nodes_v), d_child_right_v(cur-n_nodes_v)};

                // Get the split coordinate
                fp_tt const split_coord = 0.5f * (d_bboxes_v(children[0]).pmax[split_axis] + d_bboxes_v(children[1]).pmin[split_axis]);

                // Only visit the subtree containing this point
                uint32_t child_id = children[tid > split_idx];

                // Recompute min/max/is_leaf in case we changed child
                bool const is_leaf = (child_id < n_nodes_v);
                if (child_id >= n_nodes_v) {
                    range_min = d_range_min_v(child_id-n_nodes_v);
                    range_max = d_range_max_v(child_id-n_nodes_v);
                } else {
                    range_min = child_id;
                    range_max = child_id;
                }

                if (child_id == children[0]) {
                    bb.pmax[split_axis] = split_coord;
                } else {
                    bb.pmin[split_axis] = split_coord;
                }

                if (!is_leaf) {
                    // Store the corresponding bounding box 
                    d_boxes_v(child_id) = bb;

                    // Recurse if tid is in the subtree
                    cur = child_id;
                } else {  // The child is a leaf : compute
                    for (uint32_t obj_id = range_min; obj_id <= range_max; obj_id++){
                        d_boxes_v(obj_id) = bb;
                    }
                    leaf_reached = true;
                }
            }
        });

        ava_for<32>(nullptr, 0, n_nodes_v, [=] __device__ (uint32_t const tid) {

            // Get offsets of neighbors and triangles in the blocks
            uint32_t const neig_offset = tloc.get_neig_offset(tid); 
            uint32_t const elem_offset = tloc.get_elem_offset(tid); 

            uint32_t Tlast = d_node_nelemloc_v(tid);
            uint32_t cur_neig_loc = d_node_nneigloc_v(tid);

            BBoxT const bbloc = d_boxes_v(tid);

            uint32_t range_min;
            uint32_t range_max;
            Stack stack;
            stack.push(d_root_v(0));

            while (stack.len != 0) {
                uint32_t const cur = stack.peek();
                stack.pop();
                uint32_t const children[2] = {d_child_left_v(cur-n_nodes_v), d_child_right_v(cur-n_nodes_v)};

                #pragma unroll 2
                for (int ichild = 0; ichild < 2; ichild++){
                    uint32_t const child_id = children[ichild];
                    BBoxT const bbnode = d_boxes_v(child_id);

                    bool is_neighbor = true;
                    is_neighbor &= !(bbloc.max(0) < bbnode.min(0) || bbloc.min(0) > bbnode.max(0));
                    is_neighbor &= !(bbloc.max(1) < bbnode.min(1) || bbloc.min(1) > bbnode.max(1));
                    if (!is_neighbor) continue;

                    bool const is_leaf = (child_id < n_nodes_v) || (stack.len >= Stack::MaxSize);
                    if (!is_leaf) {
                        // We admit that the bounding box of the leaf is either 
                        // inside or touching the bounding box of the internal node

                        if (is_neighbor) {
                            stack.push(child_id);
                        }
                    } else {  // The child is a leaf : compute
                        if (child_id >= n_nodes_v) {
                            range_min = d_range_min_v(child_id-n_nodes_v);
                            range_max = d_range_max_v(child_id-n_nodes_v);
                        } else {
                            range_min = child_id;
                            range_max = child_id;
                        }
                        for (uint32_t obj_id = range_min; obj_id <= range_max; obj_id++){

                            // Do not check self
                            if (obj_id == tid) continue; 

                            // insert node
                            uint32_t const cur_neig_glob = obj_id;
                            tloc.set_neig(neig_offset, cur_neig_loc) = cur_neig_glob;

                            // 256-bits bitset to indicate if i-th neighbor is in the cavity
                            // of the inserted point
                            uint64_t neig_in_cavity[4] = {0, 0, 0, 0};

                            // Look at every non-delaunay elemen in the triangulation 
                            // and add it to the cavity if the inserted point is inside 
                            // its circumcircle
                            uint32_t cavity_size = 0;
                            uint32_t ti = 0;
                            while (ti < Tlast) {
                                // Get local triangle made by tid and two neighbors
                                LocalElem const elem_loc = tloc.get_elem(elem_offset, ti);

                                uint32_t i1 = tid;
                                uint32_t i2 = tloc.get_neig(neig_offset, elem_loc.a);
                                uint32_t i3 = tloc.get_neig(neig_offset, elem_loc.b);
                                uint32_t i4 = cur_neig_glob;

                                fp_tt const det = incircle_SoS(i1, i2, i3, i4, d_nodes_m_v);

                                // If the inserted point is in the circumcircle of the triangle, 
                                // add the triangle's edges to the cavity and remove it from the local 
                                // triangulation
                                if (det > 0.0f) {
                                    // Add T[ti] to the used edges
                                    neig_in_cavity[elem_loc.a >> 6] ^= 1ULL << (elem_loc.a & 63);
                                    neig_in_cavity[elem_loc.b >> 6] ^= 1ULL << (elem_loc.b & 63);
                                    cavity_size++;

                                    // Remove T[ti] from T
                                    Tlast--;
                                    tloc.get_elem(elem_offset, ti) = tloc.get_elem(elem_offset, Tlast);
                                } else {
                                    ti++;
                                }
                            }

                            if (cavity_size) {
                                // If the cavity is not empty, retriangulate the cavity
                                for (uint8_t r = 0; r < cur_neig_loc; r++){
                                    if (neig_in_cavity[r >> 6] & (1ULL << (r & 63))) {
                                        tloc.get_elem(elem_offset, Tlast) = {(uint8_t) r, (uint8_t) cur_neig_loc};
                                        Tlast++;
                                    }
                                }
                                cur_neig_loc++;
                            } 
                        }
                    }
                }
            }
            d_node_nelemloc_v(tid) = Tlast;
            d_node_nneigloc_v(tid) = cur_neig_loc;
        });
    }

    template<int dim, uint32_t block_size>
    void Mesh<dim, block_size>::insert_by_circumsphere_checking() {

        struct timespec t0, t1;

        uint32_t const n_nodes_v = n_nodes;
        uint32_t const cur_max_nneig_v = cur_max_nneig;

        TriLoc tloc = get_triloc_struct();
        AvaView<VecT, -1> d_nodes_m_v = lbvh.d_obj_m->template to_view<-1>();
        AvaView<uint32_t, -1> d_node_nelemloc_v = d_node_nelemloc->to_view<-1>();
        AvaView<uint32_t, -1> d_node_nneigloc_v = d_node_nneigloc->to_view<-1>();
        AvaView<uint8_t, -1> d_node_is_complete_v = d_node_is_complete->to_view<-1>();

        AvaView<uint32_t, -1> d_root_v = lbvh.d_root->template to_view<-1>();
        AvaView<uint32_t, -1> d_child_left_v = lbvh.d_child_left->template to_view<-1>();
        AvaView<uint32_t, -1> d_child_right_v = lbvh.d_child_right->template to_view<-1>();
        AvaView<uint32_t, -1> d_range_max_v = lbvh.d_range_max->template to_view<-1>();
        AvaView<uint32_t, -1> d_range_min_v = lbvh.d_range_min->template to_view<-1>();
        AvaView<BBoxT, -1> d_internal_data_v = lbvh.d_internal_data->template to_view<-1>();


        timespec_get(&t0, TIME_UTC);
        AvaDeviceArray<uint32_t, int>::Ptr d_node_to_insert = AvaDeviceArray<uint32_t, int>::create({(int) n_nodes_v});
        AvaDeviceArray<uint32_t, int>::Ptr d_unfinished_nodes = AvaDeviceArray<uint32_t, int>::create({(int) n_nodes_v});
        AvaDeviceArray<uint32_t, int>::Ptr d_non_delaunay_start = AvaDeviceArray<uint32_t, int>::create({(int) n_nodes_v});
        AvaDeviceArray<uint32_t, int>::Ptr d_num_selected = AvaDeviceArray<uint32_t, int>::create({1});
        timespec_get(&t1, TIME_UTC);
        printf("\tGPU memory alloc: %.5f ms\n", 
                (t1.tv_sec-t0.tv_sec)*1e3 + (t1.tv_nsec-t0.tv_nsec)*1e-6);

        AvaView<uint32_t, -1> d_node_to_insert_v = d_node_to_insert->to_view<-1>();
        AvaView<uint32_t, -1> d_unfinished_nodes_v = d_unfinished_nodes->to_view<-1>();
        AvaView<uint32_t, -1> d_non_delaunay_start_v = d_non_delaunay_start->to_view<-1>();
        ava_for<256>(nullptr, 0, n_nodes_v, [=] __device__ (uint32_t const tid) {
            d_unfinished_nodes_v(tid) = tid;
            d_non_delaunay_start_v(tid) = 0;
            d_node_is_complete_v(tid) = false;
        });

        uint32_t insert_iter = 0;
        uint32_t n_unfinished_nodes = n_nodes_v;

        while (n_unfinished_nodes && insert_iter < cur_max_nneig_v ) {
            timespec_get(&t0, TIME_UTC);
            // Find a node to insert for each local triangulation
            ava_for<256>(nullptr, 0, n_unfinished_nodes, [=] __device__ (uint32_t const tid) {
                uint32_t const node_id = d_unfinished_nodes_v(tid);
                d_node_is_complete_v(tid) = true;

                // Get offsets of neighbors and triangles in the blocks
                uint32_t const neig_offset = tloc.get_neig_offset(node_id); 
                uint32_t const elem_offset  = tloc.get_elem_offset(node_id); 

                // get the number of local element after morton neighbors insertion
                uint32_t Tlast = d_node_nelemloc_v(node_id); 
                uint32_t cur_neig_glob = node_id;

                uint32_t non_delaunay_start_idx = d_non_delaunay_start_v(node_id);
                Heap stack;

                // Get the new neighbor by finding a node inside the circumsphere 
                // of one of the current elements. If no node are found inside 
                // a given element, it is globally Delaunay.
                for (uint32_t elem_test = non_delaunay_start_idx; elem_test < Tlast; elem_test++){
                    // Get the element
                    LocalElem elem_loc = tloc.get_elem(elem_offset, elem_test);
                    uint32_t neig[2] = {
                        tloc.get_neig(neig_offset, elem_loc.a),
                        tloc.get_neig(neig_offset, elem_loc.b),
                    };

                    cur_neig_glob = node_id;

                    // Get circumsphere
                    VecT const p1 = d_nodes_m_v(node_id);
                    VecT const p2 = d_nodes_m_v(neig[0]);
                    VecT const p3 = d_nodes_m_v(neig[1]);

                    VecT const ba = p2 - p1;
                    VecT const ca = p3 - p1;
                    VecT const cb = p3 - p2;

                    fp_tt const l1 = ba.sqnorm();
                    fp_tt const l2 = ca.sqnorm();
                    fp_tt const l3 = cb.sqnorm();

                    fp_tt const det = ba.cross(ca);
                    fp_tt circum_rsqr = 1.0000001f*(
                            ( l1*l2*l3 ) / (4.f*det*det)
                            );

                    // Get circumcenter
                    fp_tt ox = p3[0] - 0.5f*(l2*cb[1] - l3*ca[1])/det;
                    fp_tt oy = p3[1] - 0.5f*(l3*ca[0] - l2*cb[0])/det;

                    // Stack for DFS on the tree
                    stack.push(d_root_v(0), 0.0f);

                    // Cannot search farther than the circumsphere
                    fp_tt best_distance = 4.f*circum_rsqr; // Can change this to get the 
                                                           // alpha-shape !

                    while (stack.len != 0) {

                        Heap::Pair const pair = stack.peek();
                        uint32_t const cur = pair.first;
                        if (pair.second >= best_distance) break;

                        stack.pop();

                        uint32_t const children[2] = {d_child_left_v(cur-n_nodes_v), d_child_right_v(cur-n_nodes_v)};

                        #pragma unroll 2
                        for (int ichild = 0; ichild < 2; ichild++){
                            uint32_t const child_id = children[ichild];
                            bool const is_leaf = (child_id < n_nodes_v);

                            if (!is_leaf) {
                                BBoxT const node_data = d_internal_data_v(child_id);

                                fp_tt dmaxx = std::fmax(p1[0] - node_data.max(0), 0.0f);
                                fp_tt dmaxy = std::fmax(p1[1] - node_data.max(1), 0.0f);
                                fp_tt dminx = std::fmax(node_data.min(0) - p1[0], 0.0f);
                                fp_tt dminy = std::fmax(node_data.min(1) - p1[1], 0.0f);
                                fp_tt const sqDistPoint = dmaxx*dmaxx + dminx*dminx + dmaxy*dmaxy + dminy*dminy; 
                                if (sqDistPoint >= best_distance) continue;

                                // Check if circumsphere intersects the internal node 
                                // And that it is closer than the current best distance.
                                dmaxx = std::fmax(ox - node_data.max(0), 0.0f);
                                dmaxy = std::fmax(oy - node_data.max(1), 0.0f);
                                dminx = std::fmax(node_data.min(0) - ox, 0.0f);
                                dminy = std::fmax(node_data.min(1) - oy, 0.0f);
                                fp_tt const sqDist = dmaxx*dmaxx + dminx*dminx + dmaxy*dmaxy + dminy*dminy; 
                                if (sqDist >= circum_rsqr) continue;

                                stack.push(child_id, sqDistPoint);
                            } else {  // The child is a leaf : compute

                                // Do not check nodes that are part of the element
                                if (child_id == node_id || child_id == neig[0] || child_id == neig[1]) continue; 

                                // Check if the node found is closer than the 
                                // current best guess and that it is inside the 
                                // circumsphere of the element
                                VecT const inserted_node = d_nodes_m_v(child_id);
                                fp_tt const d2 = (p1 - inserted_node).sqnorm();
                                if (d2 < best_distance && incircle_SoS(node_id, neig[0], neig[1], child_id, d_nodes_m_v) > 0.0f){ 
                                    cur_neig_glob = child_id;
                                    best_distance = d2;
                                }
                            }
                        }
                    }

                    if (cur_neig_glob != node_id) {
                        // If we found a node to insert, break the loop to insert it
                        break;
                    } else {
                        // If no node where found, the element is Delaunay
                        // and we can mark it as such to not use it during the 
                        // retriangulation phase
                        tloc.get_elem(elem_offset, elem_test) = std::move(tloc.get_elem(elem_offset, non_delaunay_start_idx));
                        tloc.get_elem(elem_offset, non_delaunay_start_idx) = elem_loc;
                        non_delaunay_start_idx++;
                    }
                }

                d_non_delaunay_start_v(node_id) = non_delaunay_start_idx;

                // If we didn't find a node to insert after testing all triangles 
                // then the triangulation is globally delaunay !
                if (cur_neig_glob == node_id) {
                    d_node_is_complete_v(tid) = false;
                } else {
                    d_node_to_insert_v(node_id) = cur_neig_glob;
                }
            });

            gpu_device_synchronise();
            timespec_get(&t1, TIME_UTC);
            printf("\tFind: %.5f ms\n", 
                    (t1.tv_sec-t0.tv_sec)*1e3 + (t1.tv_nsec-t0.tv_nsec)*1e-6);


            timespec_get(&t0, TIME_UTC);
            // Finished nodes do not insert anything and thus reduce the 
            // occupancy of the GPU : Compress the unfinished nodes to increase 
            // occupance on insertion
            ava::select::flagged(
                nullptr,
                temp_mem_size,
                d_unfinished_nodes->data,
                d_node_is_complete->data,
                d_num_selected->data, 
                n_unfinished_nodes
            );

            d_temp_mem->resize({temp_mem_size});

            ava::select::flagged(
                d_temp_mem->data,
                temp_mem_size,
                d_unfinished_nodes->data,
                d_node_is_complete->data,
                d_num_selected->data, 
                n_unfinished_nodes
            );

            ava::select::flagged(
                d_temp_mem->data,
                temp_mem_size,
                d_node_is_complete->data,
                d_node_is_complete->data,
                d_num_selected->data, 
                n_unfinished_nodes
            );
            gpu_memcpy(&n_unfinished_nodes, d_num_selected->data, sizeof(uint32_t), gpu_memcpy_device_to_host);

            gpu_device_synchronise();
            timespec_get(&t1, TIME_UTC);
            printf("\tCompress: %.5f ms\n", 
                    (t1.tv_sec-t0.tv_sec)*1e3 + (t1.tv_nsec-t0.tv_nsec)*1e-6);

            printf("[%u] Unfinished nodes : %u\n", insert_iter, n_unfinished_nodes);

            timespec_get(&t0, TIME_UTC);
            // Insert the node
            ava_for<32>(nullptr, 0, n_unfinished_nodes, [=] __device__ (uint32_t const tid) {
                    uint32_t const node_id = d_unfinished_nodes_v(tid);

                    // Get offsets of neighbors and triangles in the blocks
                    uint32_t const neig_offset = tloc.get_neig_offset(node_id); 
                    uint32_t const elem_offset = tloc.get_elem_offset(node_id); 

                    // get the number of local element after morton neighbors insertion
                    uint32_t Tlast = d_node_nelemloc_v(node_id); 
                    uint32_t ti = d_non_delaunay_start_v(node_id);
                    uint32_t cur_neig_glob = d_node_to_insert_v(node_id);
                    uint32_t cur_neig_loc = d_node_nneigloc_v(node_id);

                    tloc.set_neig(neig_offset, cur_neig_loc) = cur_neig_glob;

                    // 256-bits bitset to indicate if i-th neighbor is in the cavity
                    // of the inserted point
                    uint64_t neig_in_cavity[4] = {0, 0, 0, 0};

                    // Look at every non-delaunay elemen in the triangulation 
                    // and add it to the cavity if the inserted point is inside 
                    // its circumcircle
                    uint32_t cavity_size = 0;
                    while (ti < Tlast) {
                        // Get local triangle made by tid and two neighbors
                        LocalElem const elem_loc = tloc.get_elem(elem_offset, ti);

                        uint32_t i1 = node_id;
                        uint32_t i2 = tloc.get_neig(neig_offset, elem_loc.a);
                        uint32_t i3 = tloc.get_neig(neig_offset, elem_loc.b);
                        uint32_t i4 = cur_neig_glob;

                        fp_tt const det = incircle_SoS(i1, i2, i3, i4, d_nodes_m_v);

                        // If the inserted point is in the circumcircle of the triangle, 
                        // add the triangle's edges to the cavity and remove it from the local 
                        // triangulation
                        if (det > 0.0f) {
                            // Add T[ti] to the used edges
                            neig_in_cavity[elem_loc.a >> 6] ^= 1ULL << (elem_loc.a & 63);
                            neig_in_cavity[elem_loc.b >> 6] ^= 1ULL << (elem_loc.b & 63);
                            cavity_size++;

                            // Remove T[ti] from T
                            Tlast--;
                            tloc.get_elem(elem_offset, ti) = tloc.get_elem(elem_offset, Tlast);
                        } else {
                            ti++;
                        }
                    }

                    if (cavity_size) {
                        // If the cavity is not empty, retriangulate the cavity
                        for (uint8_t r = 0; r < cur_neig_loc; r++){
                            if (neig_in_cavity[r >> 6] & (1ULL << (r & 63))) {
                                tloc.get_elem(elem_offset, Tlast) = {(uint8_t) r, (uint8_t) cur_neig_loc};
                                Tlast++;
                            }
                        }
                        cur_neig_loc++;
                    } 

                    d_node_nelemloc_v(node_id) = Tlast;
                    d_node_nneigloc_v(node_id) = cur_neig_loc;
            });
            gpu_device_synchronise();
            timespec_get(&t1, TIME_UTC);
            printf("\tInsert: %.5f ms\n", 
                    (t1.tv_sec-t0.tv_sec)*1e3 + (t1.tv_nsec-t0.tv_nsec)*1e-6);


            insert_iter++;
        }
    }

    template<int dim, uint32_t block_size>
    void Mesh<dim, block_size>::remove_super_nodes(void) {

        uint32_t const n_nodes_v = n_nodes;

        TriLoc tloc = get_triloc_struct();
        AvaView<uint32_t, -1> d_node_nelemloc_v = d_node_nelemloc->to_view<-1>();
        AvaView<uint32_t, -1> d_node_nelem_out_v = d_node_nelem_out->to_view<-1>();

        ava_for<256>(nullptr, 0, n_nodes_v, [=] __device__ (uint32_t const tid) {

            // Get offsets of neighbors and triangles in the blocks
            uint32_t const neig_offset = tloc.get_neig_offset(tid); 
            uint32_t const elem_offset  = tloc.get_elem_offset(tid); 

            // get the number of local element after insertion
            uint32_t Tlast = d_node_nelemloc_v(tid); 
            uint32_t ti = 0;

            // Suppress infinity triangles and get the number of local elems 
            // and outputted global elems
            uint8_t nelem_out_loc = 0;
            uint8_t nelem_loc = 0;
            while (ti < Tlast) {
                LocalElem const elem_loc = tloc.get_elem(elem_offset, ti);

                if (elem_loc.a < n_inf_nodes || elem_loc.b < n_inf_nodes) {
                    Tlast--;
                    tloc.get_elem(elem_offset, ti) = tloc.get_elem(elem_offset, Tlast);
                    continue;
                }

                uint32_t const neig[2] = {
                    tloc.get_neig(neig_offset, elem_loc.a),
                    tloc.get_neig(neig_offset, elem_loc.b),
                };

                nelem_loc++;
                nelem_out_loc += !((tid > neig[0]) || (tid > neig[1])); // Output element if the thread is the lowest index
                ti++;
            }

            d_node_nelemloc_v(tid) = nelem_loc;
            d_node_nelem_out_v(tid+1) = nelem_out_loc;
            if (tid == 0) d_node_nelem_out_v(0) = 0;
        });
    }


    template<int dim, uint32_t block_size>
    void Mesh<dim, block_size>::compress_into_global(void) {

        // Perform a partial sum of the nelem_out array, giving the 
        // starting index of the outputted elements for each thread
        ava::scan::inplace_inclusive_sum(
            nullptr, 
            temp_mem_size,
            d_node_nelem_out->data,
            n_nodes+1
        );
        d_temp_mem->resize({temp_mem_size});
        ava::scan::inplace_inclusive_sum(
            d_temp_mem->data,
            temp_mem_size, 
            d_node_nelem_out->data,
            n_nodes+1
        );

        gpu_memcpy(&n_elems, d_node_nelem_out->data + n_nodes, sizeof(n_elems), gpu_memcpy_device_to_host);
        d_elemglob->resize({(int) n_elems});

        TriLoc const tloc = get_triloc_struct();
        AvaView<uint32_t, -1> d_node_nelem_out_v = d_node_nelem_out->to_view<-1>();
        AvaView<Elem, -1> d_elemglob_v = d_elemglob->template to_view<-1>();
        ava_for<256>(nullptr, 0, n_nodes, [=] __device__(uint32_t const tid) {
            uint32_t const neig_offset = tloc.get_neig_offset(tid);
            uint32_t const elem_offset = tloc.get_elem_offset(tid); 
            uint32_t triIdx = 0;
            for (uint32_t i = d_node_nelem_out_v(tid); i < d_node_nelem_out_v(tid+1); i++){
                LocalElem tricur;
                uint32_t neig[2];
                do {
                    tricur = tloc.get_elem(elem_offset, triIdx);
                    neig[0] = tloc.get_neig(neig_offset, tricur.a); 
                    neig[1] = tloc.get_neig(neig_offset, tricur.b);
                    triIdx++;
                } while (neig[0] < tid || neig[1] < tid);

                d_elemglob_v(i) = {tid, neig[0], neig[1]};
            }
        });
    }

} // namespace stream::mesh


#ifdef __cplusplus
extern "C" {
#endif

// Create an empty 2D mesh
Mesh2D* Mesh2D_create() {
    return new Mesh2D;
}

// Destroy a 2D mesh
void Mesh2D_destroy(Mesh2D * mesh){
    delete mesh;
}

// Set the nodes of the 2D mesh
void Mesh2D_set_nodes(Mesh2D * const mesh, uint32_t nnodes, fp_tt const * const nodes) {
    using VecT = Mesh2D::VecT;
    AvaDeviceArray<VecT, int>::Ptr d_nodes = AvaDeviceArray<VecT, int>::create({(int) nnodes});

    gpu_memcpy(d_nodes->data, nodes, sizeof(VecT)*nnodes, gpu_memcpy_host_to_device);

    mesh->d_nodes = d_nodes;
}

// Initialize the mesh by inserting the root node in the super-simplex
void Mesh2D_init(Mesh2D * const mesh) {
    mesh->init();
}

// Insert a few morton neighbors in the local triangulation
void Mesh2D_insert_morton_neighbors(Mesh2D * const mesh) {
    mesh->insert_morton_neighbors();
}

// Insert 2 nodes per quadrant around the root node in the local triangulation
void Mesh2D_insert_quadrant_neighbors(Mesh2D * const mesh) {
    mesh->insert_quadrant_neighbors();
}

// Insert all the leaves of the BVH that are adjaccent to the leaf of 
// the root node
void Mesh2D_insert_BVH_neighbors(Mesh2D * const mesh) {
    mesh->insert_BVH_neighbors();
}

void Mesh2D_remove_super_nodes(Mesh2D * const mesh) {
    mesh->remove_super_nodes();
}

void Mesh2D_insert_iterative(Mesh2D * const mesh) {
    mesh->insert_by_circumsphere_checking();
}

uint32_t Mesh2D_get_nelem(Mesh2D * const mesh) {
    mesh->compress_into_global();
    return mesh->n_elems;
}

void Mesh2D_get_elem(Mesh2D * const mesh, uint32_t * const elems) {
    using Elem = Mesh2D::Elem;

    gpu_memcpy(elems, mesh->d_elemglob->data, sizeof(Elem)*mesh->n_elems, gpu_memcpy_device_to_host);
}

void Mesh2D_get_ordered_nodes(Mesh2D * const mesh, fp_tt * const nodes) {
    gpu_memcpy(nodes, mesh->lbvh.d_obj_m->data, sizeof(Mesh2D::VecT)*mesh->n_nodes, gpu_memcpy_device_to_host);
}


#ifdef __cplusplus
}
#endif
