#ifndef __STREAM_TRBVH_HPP__
#define __STREAM_TRBVH_HPP__

/*
 *  Implementation of the Linear Bounding Volume Hierarchy (LBVH) from :
 *     C. Apetrei 2014, Eurographics. 
 *     "Fast and Simple Agglomerative LBVH construction"
 */

#include <cfloat>
#include <utility>
#include "ava_device_array.h"
#include "ava_device_array.hpp"
#include "primitives.hpp"
#include "vec.hpp"
#include "ava_reduce.h"
#include "ava_view.h"
#include "ava_atomic.h"
#include "ava_sort.h"
#include "morton.hpp"

// Forward declarations
namespace stream::geo {
    template<typename ObjT, typename DataF, int dim> struct TRBVH;
} // namespace stream::geo

// Typedefs for often used template arguments
#ifdef __cplusplus
extern "C" {
#endif

    typedef stream::geo::TRBVH<Vec2f, stream::geo::ComputeBB_Functor<Vec2f>, 2> Point2DTRBVH;
    typedef stream::geo::TRBVH<Sphere2D, stream::geo::ComputeBB_Functor<Sphere2D>, 2> Sphere2DTRBVH;
    typedef stream::geo::TRBVH<Edge2D, stream::geo::ComputeBB_Functor<Edge2D>, 2> Edge2DTRBVH;
    typedef stream::geo::TRBVH<Tri2D, stream::geo::ComputeBB_Functor<Tri2D>, 2> Tri2DTRBVH;

    typedef stream::geo::TRBVH<Sphere3D, stream::geo::ComputeBB_Functor<Sphere3D>, 3> Sphere3DTRBVH;
    
#ifdef __cplusplus
}
#endif

namespace stream::geo {

    template<typename ObjT, typename DataF, int dim>
    struct TRBVH {
        using Scalar = typename ObjT::Scalar;
        using DataComputeT = decltype(std::declval<DataF>().init(ObjT()));
        using DataT = decltype(std::declval<DataF>().finalize(DataComputeT()));
        using VecT = Vec<Scalar, dim>;
        using BBoxT = BBox<Scalar, dim>;

        static constexpr fp_tt const Ci = 1.2f;
        static constexpr fp_tt const Ct = 1.0f;
        static constexpr uint8_t const treelet_size = 5;
        static constexpr uint32_t const gamma_init = treelet_size;

        uint32_t n; // number of objects
                    
        size_t tmp_size; // Size of temporary storage in bytes
        AvaDeviceArray<char, size_t>::Ptr tmp; // Temporary storage
                                 
        typename AvaDeviceArray<BBoxT, int>::Ptr d_bb_glob; // BBox of all objects
        AvaDeviceArray<uint32_t, int>::Ptr d_root; // Index of the root node

        AvaDeviceArray<uint64_t, int>::Ptr d_morton;     // d_morton[i] = morton(d_coords[i])
        AvaDeviceArray<uint64_t, int>::Ptr d_morton_sorted; // Sorted morton codes

        AvaDeviceArray<uint32_t, int>::Ptr d_map;         // 0, 1, 2, 3, ..., n-1
        AvaDeviceArray<uint32_t, int>::Ptr d_map_sorted;  // d_map sorted accorting to the morton codes

        AvaDeviceArray<uint32_t, int>::Ptr d_child_left;   // ID of left child
        AvaDeviceArray<uint32_t, int>::Ptr d_child_right;  // ID of right child
        AvaDeviceArray<uint32_t, int>::Ptr d_range_min;    // starting index of the range of leaves in the subtree of i-th internal node
        AvaDeviceArray<uint32_t, int>::Ptr d_range_max;    // last index of the range of leaves in the subtree of i-th internal node
        AvaDeviceArray<uint64_t, int>::Ptr d_delta;  // Delta function
                    
        AvaDeviceArray<fp_tt, int>::Ptr d_sah_cost; 
        AvaDeviceArray<uint32_t, int>::Ptr d_subtree_size; 

        typename AvaDeviceArray<DataT, int>::Ptr d_internal_data;  // Data stored on each internal node
        typename AvaDeviceArray<BBoxT, int>::Ptr d_bboxes;  // Bboxes of all nodes
                                  
        typename AvaDeviceArray<ObjT, int>::Ptr d_obj;   // Input objects : store only the pointer.
        typename AvaDeviceArray<ObjT, int>::Ptr d_obj_m; // objects sorted according to morton codes
                                   
        AvaDeviceArray<uint32_t, int>::Ptr d_touched; // Has the node already been touched (for the data bottom-up parallel compute)
        AvaDeviceArray<uint32_t, int>::Ptr d_parent; // The parent of the nodes. 
                                                     // The array has size 2n - 1 :
                                                     //   n leaf nodes, then n-1 internal nodes

        TRBVH() {
            n = 0;
            tmp_size = 0;

            tmp = AvaDeviceArray<char, size_t>::create({0});
            d_bb_glob = AvaDeviceArray<BBoxT, int>::create({1});
            d_bboxes = AvaDeviceArray<BBoxT, int>::create({0});
            d_root = AvaDeviceArray<uint32_t, int>::create({1});

            d_morton = AvaDeviceArray<uint64_t, int>::create({0});
            d_morton_sorted = AvaDeviceArray<uint64_t, int>::create({0});

            d_map = AvaDeviceArray<uint32_t, int>::create({0});
            d_map_sorted = AvaDeviceArray<uint32_t, int>::create({0});

            d_child_left = AvaDeviceArray<uint32_t, int>::create({0});
            d_child_right = AvaDeviceArray<uint32_t, int>::create({0});
            d_range_min = AvaDeviceArray<uint32_t, int>::create({0});
            d_range_max = AvaDeviceArray<uint32_t, int>::create({0});
            d_delta = AvaDeviceArray<uint64_t, int>::create({0});

            d_internal_data = AvaDeviceArray<DataT, int>::create({0});
            d_obj_m = AvaDeviceArray<ObjT, int>::create({0});

            d_touched = AvaDeviceArray<uint32_t, int>::create({0});
            d_parent = AvaDeviceArray<uint32_t, int>::create({0});

            d_sah_cost = AvaDeviceArray<fp_tt, int>::create({0});
            d_subtree_size = AvaDeviceArray<uint32_t, int>::create({0});
        }

        void set_objects(typename AvaDeviceArray<ObjT, int>::Ptr objects) {
            n = objects->size;

            d_morton = AvaDeviceArray<uint64_t, int>::create({(int) n});
            d_morton_sorted = AvaDeviceArray<uint64_t, int>::create({(int) n});

            d_map = AvaDeviceArray<uint32_t, int>::create({(int) n});
            d_map_sorted = AvaDeviceArray<uint32_t, int>::create({(int) n});

            d_child_left = AvaDeviceArray<uint32_t, int>::create({(int) (n-1)});
            d_child_right = AvaDeviceArray<uint32_t, int>::create({(int) (n-1)});
            d_range_min = AvaDeviceArray<uint32_t, int>::create({(int) (n-1)});
            d_range_max = AvaDeviceArray<uint32_t, int>::create({(int) (n-1)});
            d_delta = AvaDeviceArray<uint64_t, int>::create({(int) (n-1)});

            d_internal_data = AvaDeviceArray<DataT, int>::create({(int) (2*n-1)});
            d_obj_m = AvaDeviceArray<ObjT, int>::create({(int) n});
            d_obj = objects; // copy pointer

            d_touched = AvaDeviceArray<uint32_t, int>::create({(int) (n-1)});
            d_parent = AvaDeviceArray<uint32_t, int>::create({(int) (2*n-1)});

            d_sah_cost = AvaDeviceArray<fp_tt, int>::create({(int) (2*n-1)});
            d_subtree_size = AvaDeviceArray<uint32_t, int>::create({(int) (2*n-1)});
            d_bboxes = AvaDeviceArray<BBoxT, int>::create({(int) (2*n-1)});
        }

        // Compute the bounding box of the set of objects
        void _compute_global_bbox() {
            constexpr BBoxT bb_data = BBoxT::empty();

            const auto transform_op = [=] __device__ (const ObjT& a) -> BBoxT {
                return a.get_bbox(); 
            };

            const auto reduce_bbox_op = [=] __device__ (const BBoxT& a, const BBoxT& b) -> BBoxT {
                BBoxT ret = a;
                ret.combineBox(b);
                return ret;
            };

            ava::reduce::transform_reduce(
                nullptr,
                tmp_size,
                d_obj->data,
                d_bb_glob->data,
                n,
                reduce_bbox_op,
                transform_op,
                bb_data);

            tmp->resize({tmp_size});

            ava::reduce::transform_reduce(
                tmp->data,
                tmp_size,
                d_obj->data,
                d_bb_glob->data,
                n,
                reduce_bbox_op,
                transform_op,
                bb_data);
        }

        // Compute the morton codes of the centroid of each object
        void _compute_morton_codes() {

            AvaView<ObjT, -1>     d_obj_v     = d_obj->template to_view<-1>();
            AvaView<BBoxT, -1>    d_bb_glob_v = d_bb_glob->template to_view<-1>();
            AvaView<uint32_t, -1> d_map_v     = d_map->to_view<-1>();
            AvaView<uint64_t, -1> d_morton_v  = d_morton->to_view<-1>();

            ava_for<AVA_BLOCK_SIZE>(0, 0, n, [=] __device__ (size_t const tid){
                constexpr uint64_t imax = 1ULL << ((8*sizeof(imax)) / dim);
                VecT const range = d_bb_glob_v(0).pmax - d_bb_glob_v(0).pmin;
                VecT mortonFactor;
                for (int i = 0; i < dim; ++i){
                    Scalar f = imax / range[i];
                    if (range[i]*f >= static_cast<Scalar>(imax)) f = std::nextafter(f, static_cast<Scalar>(0));
                    mortonFactor[i] = f;
                }

                uint64_t morton;
                VecT const c = cMult(d_obj_v(tid).get_centroid() - d_bb_glob_v(0).pmin, mortonFactor);
                if constexpr (dim == 2) {
                    morton = stream::geo::encode_magicbits2D(
                            static_cast<uint32_t>(c[0]), 
                            static_cast<uint32_t>(c[1])
                        );
                } else {
                    morton = stream::geo::encode_magicbits3D(
                            static_cast<uint32_t>(c[0]),
                            static_cast<uint32_t>(c[1]), 
                            static_cast<uint32_t>(c[2])
                        );
                }

                d_map_v(tid) = tid;
                d_morton_v(tid) = morton;
            });
        }

        // Sort the objects according to their morton code
        void _sort_objects() {
            ava::sort::sort_pairs(
                nullptr, 
                tmp_size,
                d_morton->data,
                d_morton_sorted->data,
                d_map->data,
                d_map_sorted->data,
                n);

            tmp->resize({tmp_size});

            ava::sort::sort_pairs(
                tmp->data, 
                tmp_size,
                d_morton->data,
                d_morton_sorted->data,
                d_map->data,
                d_map_sorted->data,
                n);

            AvaView<ObjT, -1>     d_obj_v        = d_obj->template to_view<-1>();
            AvaView<ObjT, -1>     d_obj_m_v      = d_obj_m->template to_view<-1>();
            AvaView<uint32_t, -1> d_map_sorted_v = d_map_sorted->to_view<-1>();
            AvaView<uint64_t, -1> d_delta_v      = d_delta->to_view<-1>();
            AvaView<uint64_t, -1> d_morton_sorted_v = d_morton_sorted->to_view<-1>();
            AvaView<DataT, -1>    d_internal_data_v = d_internal_data->template to_view<-1>();
            AvaView<uint32_t, -1> d_touched_v       = d_touched->to_view<-1>();
            AvaView<uint32_t, -1> d_subtree_size_v  = d_subtree_size->to_view<-1>();
            AvaView<BBoxT, -1>    d_bboxes_v        = d_bboxes->template to_view<-1>();
            uint32_t const n_v = n;
            ava_for<AVA_BLOCK_SIZE>(0, 0, n_v, [=] __device__ (size_t const tid) {
                d_obj_m_v(tid) = d_obj_v(d_map_sorted_v(tid));

                // At the same time, init the leaf nodes
                DataF const functor;
                ObjT const p = d_obj_m_v(tid);
                DataComputeT const data = functor.init(p);
                d_internal_data_v(tid) = data;
                d_bboxes_v(tid) = p.get_bbox();

                // And compute the deltas
                if (tid < n_v-1) {
                    d_delta_v(tid) = d_morton_sorted_v(tid) ^ d_morton_sorted_v(tid+1);

                    // And reset the touched flag 
                    d_touched_v(tid) = 0;
                }

                d_subtree_size_v(tid) = 1;
            });
        }

        // Build the hierachy of the LBVH
        void _build_hierarchy() {
            AvaView<uint64_t, -1> d_delta_v         = d_delta->to_view<-1>();
            AvaView<DataT, -1>    d_internal_data_v = d_internal_data->template to_view<-1>();
            AvaView<uint32_t, -1> d_touched_v       = d_touched->to_view<-1>();
            AvaView<uint32_t, -1> d_parent_v        = d_parent->to_view<-1>();
            AvaView<uint32_t, -1> d_range_min_v     = d_range_min->to_view<-1>();
            AvaView<uint32_t, -1> d_range_max_v     = d_range_max->to_view<-1>();
            AvaView<uint32_t, -1> d_child_left_v    = d_child_left->to_view<-1>();
            AvaView<uint32_t, -1> d_child_right_v   = d_child_right->to_view<-1>();
            AvaView<uint32_t, -1> d_root_v          = d_root->to_view<-1>();
            AvaView<uint32_t, -1> d_subtree_size_v  = d_subtree_size->to_view<-1>();
            AvaView<BBoxT, -1>    d_bb_glob_v       = d_bb_glob->template to_view<-1>();
            AvaView<ObjT, -1>     d_obj_m_v         = d_obj_m->template to_view<-1>();
            AvaView<BBoxT, -1>    d_bboxes_v        = d_bboxes->template to_view<-1>();
            AvaView<fp_tt, -1>    d_sah_cost_v      = d_sah_cost->to_view<-1>();
            uint32_t n_v = static_cast<uint32_t>(n);

            ava_for<AVA_BLOCK_SIZE>(0, 0, n_v, [=] __device__ (uint32_t const i) {
                uint32_t idx = i;

                DataF functor;
                DataT data = d_internal_data_v(i);
                uint32_t imin = i;
                uint32_t imax = i;
                uint32_t parent;
                uint32_t subtree_size = 1;

                // SAH cost
                BBoxT cur_bb = d_obj_m_v(idx).get_bbox();
                VecT const obj_ext = cur_bb.pmax - cur_bb.pmin;
                fp_tt inner_cost = Ct * obj_ext[0] * obj_ext[1];

                d_sah_cost_v(idx) = 0.0f; 

                while (true) {
                    // We are at the root
                    if (imin == 0 && imax == n_v-1) {
                        printf("SAH cost after building : %.5f\n", d_sah_cost_v(idx));
                        d_parent_v(idx) = 2*n_v;
                        d_root_v(0) = idx;
                        break;
                    }

                    bool const right_parent = imin == 0 || (imax != n_v-1 && d_delta_v(imax) < d_delta_v(imin-1));
                    if (right_parent) {
                        // Attach node to right parent
                        parent = imax + n_v;
                        d_child_left_v(parent-n_v) = idx;
                        d_range_min_v(parent-n_v) = imin;
                        __threadfence();
                    } else {
                        // Attach node to left parent
                        parent = imin + n_v - 1;
                        d_child_right_v(parent-n_v) = idx;
                        d_range_max_v(parent-n_v) = imax;
                        __threadfence();
                    }
                    d_parent_v(idx) = parent;

                    if (ava::atomic::fetch_add(&d_touched_v(parent-n_v), 1U)) {
                        uint32_t sibling;
                        if (right_parent) {
                            sibling = d_child_right_v(parent-n_v);
                        } else {
                            sibling = d_child_left_v(parent-n_v);
                        }

                        DataT sibling_data = d_internal_data_v(sibling);
                        subtree_size += d_subtree_size_v(sibling);
                        functor.combine(data, sibling_data);

                        BBoxT const sibling_bb = d_bboxes_v(sibling);
                        cur_bb.combineBox(sibling_bb);

                        VecT const node_extents = cur_bb.pmax - cur_bb.pmin;
                        inner_cost += Ci * node_extents[0]*node_extents[1] + d_sah_cost_v(sibling);

                        d_sah_cost_v(parent) = inner_cost;
                        d_subtree_size_v(parent) = subtree_size;
                        d_internal_data_v(parent) = data;
                        d_bboxes_v(parent) = cur_bb;
                        __threadfence();

                        idx = parent;
                        if (right_parent) imax = d_range_max_v(parent-n_v);
                        else              imin = d_range_min_v(parent-n_v);
                    } else {
                        // Terminate thread
                        break;
                    }
                }
            });
        }

        void treelet_optimize(uint32_t const min_size) {
            d_touched->set_scalar(0);

            AvaView<DataT, -1>    d_internal_data_v = d_internal_data->template to_view<-1>();
            AvaView<uint32_t, -1> d_touched_v       = d_touched->to_view<-1>();
            AvaView<uint32_t, -1> d_parent_v        = d_parent->to_view<-1>();
            AvaView<uint32_t, -1> d_child_left_v    = d_child_left->to_view<-1>();
            AvaView<uint32_t, -1> d_child_right_v   = d_child_right->to_view<-1>();
            AvaView<uint32_t, -1> d_subtree_size_v  = d_subtree_size->to_view<-1>();
            AvaView<BBoxT, -1>    d_bboxes_v        = d_bboxes->template to_view<-1>();
            AvaView<ObjT, -1>     d_obj_m_v         = d_obj_m->template to_view<-1>();
            AvaView<fp_tt, -1>    d_sah_cost_v      = d_sah_cost->to_view<-1>();
            uint32_t const n_v = static_cast<uint32_t>(n);
            uint32_t const min_size_v = min_size;


            ava_for<AVA_BLOCK_SIZE>(0, 0, n_v, [=] __device__ (uint32_t const i) {
                uint32_t idx = i;
                uint32_t parent = d_parent_v(i);
                uint32_t subtree_size = d_subtree_size_v(i);

                while (true) {
                    if (subtree_size >= min_size_v){ 
                        
                        // Inner nodes of the treelet
                        uint32_t treelet_inner[treelet_size-1] = {0}; 
                        // Leaves of the treelet
                        uint32_t treelet_leaves[treelet_size] = {0};
                        // SAH cost of the treelet's leaves
                        fp_tt treelet_cost[treelet_size] = {0};

                        // ===================== Find a treelet ================

                        treelet_inner[0] = idx; // Init with current root
                        treelet_leaves[0] = d_child_left_v(idx-n_v); // add the 2 children
                        treelet_leaves[1] = d_child_right_v(idx-n_v);
                        treelet_cost[0] = d_sah_cost_v(treelet_leaves[0]);
                        treelet_cost[1] = d_sah_cost_v(treelet_leaves[1]);

                        for (uint8_t inner = 1; inner < treelet_size-1; inner++){
                            fp_tt best_cost = treelet_cost[0];
                            uint32_t best_leaf_idx = 0;

                            // Find the leaf with the highest SAH
                            for (uint8_t leaf = 1; leaf < inner+1; leaf++){
                                if (treelet_cost[leaf] > best_cost){
                                    best_cost = treelet_cost[leaf];
                                    best_leaf_idx = leaf;
                                }
                            }

                            // Transform the leaf with highest SAH as an 
                            // internal node of the treelet
                            uint32_t best_leaf = treelet_leaves[best_leaf_idx];
                            treelet_inner[inner] = best_leaf;
                            treelet_leaves[best_leaf_idx] = treelet_leaves[inner];
                            treelet_cost[best_leaf_idx] = treelet_cost[inner];

                            treelet_leaves[inner] = d_child_left_v(best_leaf-n_v);
                            treelet_cost[inner] = d_sah_cost_v(treelet_leaves[inner]);

                            treelet_leaves[inner+1] = d_child_right_v(best_leaf-n_v);
                            treelet_cost[inner+1] = d_sah_cost_v(treelet_leaves[inner+1]);
                        }

                        // =============== Optimize the treelet ===============

                        // Compute surface area of each subset
                        constexpr uint8_t num_subsets = 1 << treelet_size;
                        fp_tt areas[(1 << treelet_size)] = {0};
                        // binary representation of s gives the partitioning of the 
                        // treelet leaves
                        for (uint8_t s = 1; s < num_subsets; s++) {
                            BBoxT bbp = BBoxT::empty();
                            for (uint8_t l = 0; l < treelet_size; l++){
                                if (s & (1u << l)) {
                                    bbp.combineBox(d_bboxes_v(treelet_leaves[l]));
                                }
                            }
                            VecT const bbp_extents = bbp.pmax - bbp.pmin;
                            areas[s] = bbp_extents[0] * bbp_extents[1];
                        }

                        // Initialize cost of individual leaves
                        fp_tt c_opt[num_subsets] = {0};
                        for (uint8_t i = 0; i < treelet_size; i++){
                            c_opt[1<<i] = treelet_cost[i];
                        }

                        // Optimize subset 
                        uint8_t p_opt[num_subsets] = {0};
                        for (uint8_t k = 2; k <= treelet_size; k++) {
                            for (uint8_t s = 1; s < num_subsets; s++){
                                if (__builtin_popcount(s) != k) continue;

                                fp_tt cs = FLT_MAX;
                                uint8_t ps = 0;

                                uint8_t delta = (s - 1) & s;
                                uint8_t p = (-delta) & s;
                                while (p != 0) {
                                    fp_tt c = c_opt[p] + c_opt[s ^ p];
                                    if (c < cs) {
                                        cs = c;
                                        ps = p;
                                    }
                                    p = (p - delta) & s;
                                }

                                uint32_t t = 0;
                                for (uint8_t l = 0; l < treelet_size; l++){
                                    if (s & (1u << l)) {
                                        t += d_subtree_size_v(treelet_leaves[l]);
                                    }
                                }

                                // Compute final SAH 
                                c_opt[s] = std::fmin(Ci*areas[s] + cs, Ct*areas[s]*t);
                                p_opt[s] = ps;
                            }
                        }

                        // ===================== Restructure tree =============
                        //   The array c_opt contains the optimal SAH cost
                        //   The array p_opt contains the optimal partioning
                        //
                        //   We need to backtrack from p_opt[num_subsets-1]
                        DataF const functor;
                        constexpr uint8_t const mask = num_subsets - 1;
                        uint8_t index = 0;
                        uint8_t stack_partition[treelet_size] = {0};
                        uint8_t stack_left[treelet_size] = {0};
                        uint32_t stack_parent[treelet_size] = {0};
                        uint8_t stack_size = 0;

                        d_sah_cost_v(idx) = c_opt[mask];

                        // Left subtree
                        stack_partition[0] = p_opt[mask];
                        stack_left[0] = true;
                        stack_parent[0] = treelet_inner[index];
                        stack_size++;

                        // Right subtree
                        stack_partition[1] = (~p_opt[mask]) & mask;
                        stack_left[1] = false;
                        stack_parent[1] = treelet_inner[index];
                        stack_size++;


                        while (stack_size != 0) {
                            stack_size--;
                            uint8_t partition = stack_partition[stack_size];
                            uint8_t is_left = stack_left[stack_size];
                            uint32_t parent_loc = stack_parent[stack_size];

                            bool const is_leaf = __builtin_popcount(partition) == 1;
                            uint32_t node_idx;

                            if (is_leaf) {
                                // Leaf node
                                node_idx = treelet_leaves[__builtin_ffs(partition)-1];
                            } else {
                                // Internal node
                                node_idx = treelet_inner[++index];
                            }

                            if (is_left) {
                                d_child_right_v(parent_loc-n_v) = node_idx;
                            } else {
                                d_child_left_v(parent_loc-n_v) = node_idx;
                            }

                            d_parent_v(node_idx) = parent_loc;
                            __threadfence();

                            if (!is_leaf) {
                                // Recompute bboxes based on leaves in the subtree
                                BBoxT bbloc = BBoxT::empty();
                                DataT data;
                                uint32_t sub_size = 0;
                                bool data_is_init = false;

                                for (uint8_t l = 0; l < treelet_size; l++){
                                    if (partition & (1u << l)) {
                                        bbloc.combineBox(d_bboxes_v(treelet_leaves[l]));
                                        sub_size += d_subtree_size_v(treelet_leaves[l]);

                                        if (data_is_init) {
                                            functor.combine(data, d_internal_data_v(treelet_leaves[l]));
                                        } else {
                                            data = d_internal_data_v(treelet_leaves[l]);
                                            data_is_init = true;
                                        }
                                    }
                                }
                                d_bboxes_v(node_idx) = bbloc;
                                d_internal_data_v(node_idx) = data;
                                d_subtree_size_v(node_idx) = sub_size;
                                d_sah_cost_v(node_idx) = c_opt[partition];
                                __threadfence();

                                uint8_t partition_left = p_opt[partition];
                                uint8_t partition_right = (~partition_left) & partition;

                                stack_partition[stack_size] = partition_left;
                                stack_left[stack_size] = true;
                                stack_parent[stack_size] = node_idx;
                                stack_size++;

                                stack_partition[stack_size] = partition_right;
                                stack_left[stack_size] = false;
                                stack_parent[stack_size] = node_idx;
                                stack_size++;
                            }
                        }
                    }
                    
                    if (parent == 2*n_v) break; // If we reached the root, stop
                    if (ava::atomic::fetch_add(&d_touched_v(parent-n_v), 1U)) {
                        // This thread goes up to the parent
                        idx = parent;
                        subtree_size = d_subtree_size_v(idx);
                        parent = d_parent_v(idx);
                    } else {
                        // This thread terminates
                        break;
                    }
                }
            });
        }


        // Build the LBVH tree
        void build(){
            struct timespec t0, t1;
            printf("================ Tree Construction ====================\n");

            timespec_get(&t0, TIME_UTC);
            _compute_global_bbox();
            gpu_device_synchronise();
            timespec_get(&t1, TIME_UTC);
            printf("Compute global bbox : %.5f ms\n",
                    (t1.tv_sec - t0.tv_sec)*1e3 + (t1.tv_nsec - t0.tv_nsec)*1e-6);

            timespec_get(&t0, TIME_UTC);
            _compute_morton_codes();
            gpu_device_synchronise();
            timespec_get(&t1, TIME_UTC);
            printf("Compute morton codes: %.5f ms\n",
                    (t1.tv_sec - t0.tv_sec)*1e3 + (t1.tv_nsec - t0.tv_nsec)*1e-6);

            timespec_get(&t0, TIME_UTC);
            _sort_objects();
            gpu_device_synchronise();
            timespec_get(&t1, TIME_UTC);
            printf("Sort objects according to morton codes: %.5f ms\n",
                    (t1.tv_sec - t0.tv_sec)*1e3 + (t1.tv_nsec - t0.tv_nsec)*1e-6);

            timespec_get(&t0, TIME_UTC);
            _build_hierarchy();
            gpu_device_synchronise();
            timespec_get(&t1, TIME_UTC);
            printf("Build hierarchy (build+fit): %.5f ms\n",
                    (t1.tv_sec - t0.tv_sec)*1e3 + (t1.tv_nsec - t0.tv_nsec)*1e-6);

            timespec_get(&t0, TIME_UTC);
            uint32_t min_size_cur = gamma_init;
            for (int i = 0; i < 3; ++i){
                treelet_optimize(min_size_cur);
                min_size_cur *= 2;
            }
            gpu_device_synchronise();
            timespec_get(&t1, TIME_UTC);
            printf("Optimize hierarchy : %.5f ms\n",
                    (t1.tv_sec - t0.tv_sec)*1e3 + (t1.tv_nsec - t0.tv_nsec)*1e-6);
        }

    };

} // namespace stream::geo

#endif // __STREAM_TRBVH_HPP__

