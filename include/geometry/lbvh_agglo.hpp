#ifndef __STREAM_LBVH_AGGLO_HPP__
#define __STREAM_LBVH_AGGLO_HPP__

/*
 *  Implementation of the Linear Bounding Volume Hierarchy (LBVH) from :
 *     C. Apetrei 2014, Eurographics. 
 *     "Fast and Simple Agglomerative LBVH construction"
 */

#include <utility>
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
    template<typename ObjT, typename DataF, int dim> struct LBVHa;
} // namespace stream::geo

// Typedefs for often used template arguments
#ifdef __cplusplus
extern "C" {
#endif

    typedef stream::geo::LBVHa<Vec2f, stream::geo::ComputeBB_Functor<Vec2f>, 2> Point2DLBVHa;
    typedef stream::geo::LBVHa<Sphere2D, stream::geo::ComputeBB_Functor<Sphere2D>, 2> Sphere2DLBVHa;
    typedef stream::geo::LBVHa<Edge2D, stream::geo::ComputeBB_Functor<Edge2D>, 2> Edge2DLBVHa;
    typedef stream::geo::LBVHa<Tri2D, stream::geo::ComputeBB_Functor<Tri2D>, 2> Tri2DLBVHa;

    typedef stream::geo::LBVHa<Sphere3D, stream::geo::ComputeBB_Functor<Sphere3D>, 3> Sphere3DLBVHa;
    
#ifdef __cplusplus
}
#endif

namespace stream::geo {

    template<typename ObjT, typename DataF, int dim>
    struct LBVHa {
        using Scalar = typename ObjT::Scalar;
        using DataComputeT = decltype(std::declval<DataF>().init(ObjT()));
        using DataT = decltype(std::declval<DataF>().finalize(DataComputeT()));
        using VecT = Vec<Scalar, dim>;
        using BBoxT = BBox<Scalar, dim>;

        uint32_t n; // number of objects
                    
        size_t tmp_size; // Size of temporary storage in bytes
        AvaDeviceArray<char, size_t>::Ptr tmp; // Temporary storage
                                 
        typename AvaDeviceArray<BBoxT, int>::Ptr d_bb_glob; // BBox of all objects
        AvaDeviceArray<uint32_t, int>::Ptr d_root; // Index of the root node

        AvaDeviceArray<uint64_t, int>::Ptr d_morton;     // d_morton[i] = morton(d_coords[i])
        AvaDeviceArray<uint64_t, int>::Ptr d_morton_sorted; // Sorted morton codes

        AvaDeviceArray<uint32_t, int>::Ptr d_map;         // 0, 1, 2, 3, ..., n-1
        AvaDeviceArray<uint32_t, int>::Ptr d_map_sorted;  // d_map sorted accorting to the morton codes

        AvaDeviceArray<uint32_t, int>::Ptr d_children;     // ID of children 
        AvaDeviceArray<uint32_t, int>::Ptr d_range;        // Range of indices of leaves in the subtree of i-th internal node
        AvaDeviceArray<uint64_t, int>::Ptr d_delta;        // Delta function
        AvaDeviceArray<uint32_t, int>::Ptr d_split_idx;    // Axis of the split

        typename AvaDeviceArray<DataT, int>::Ptr d_internal_data;  // Data stored on each internal node
                                  
        typename AvaDeviceArray<ObjT, int>::Ptr d_obj;   // Input objects : store only the pointer.
        typename AvaDeviceArray<ObjT, int>::Ptr d_obj_m; // objects sorted according to morton codes
                                   
        AvaDeviceArray<uint32_t, int>::Ptr d_touched; // Has the node already been touched (for the data bottom-up parallel compute)
        AvaDeviceArray<uint32_t, int>::Ptr d_parent; // The parent of the nodes. 
                                                     // The array has size 2n - 1 :
                                                     //   n leaf nodes, then n-1 internal nodes

        LBVHa() {
            n = 0;
            tmp_size = 0;

            tmp = AvaDeviceArray<char, size_t>::create({0});
            d_bb_glob = AvaDeviceArray<BBoxT, int>::create({1});
            d_root = AvaDeviceArray<uint32_t, int>::create({1});

            d_morton = AvaDeviceArray<uint64_t, int>::create({0});
            d_morton_sorted = AvaDeviceArray<uint64_t, int>::create({0});

            d_map = AvaDeviceArray<uint32_t, int>::create({0});
            d_map_sorted = AvaDeviceArray<uint32_t, int>::create({0});

            d_children = AvaDeviceArray<uint32_t, int>::create({0, 2});
            d_range = AvaDeviceArray<uint32_t, int>::create({0, 2});
            d_delta = AvaDeviceArray<uint64_t, int>::create({0});
            d_split_idx = AvaDeviceArray<uint32_t, int>::create({0});

            d_internal_data = AvaDeviceArray<DataT, int>::create({0});
            d_obj_m = AvaDeviceArray<ObjT, int>::create({0});

            d_touched = AvaDeviceArray<uint32_t, int>::create({0});
            d_parent = AvaDeviceArray<uint32_t, int>::create({0});
        }

        void set_objects(typename AvaDeviceArray<ObjT, int>::Ptr objects) {
            n = objects->size;

            d_morton = AvaDeviceArray<uint64_t, int>::create({(int) n});
            d_morton_sorted = AvaDeviceArray<uint64_t, int>::create({(int) n});

            d_map = AvaDeviceArray<uint32_t, int>::create({(int) n});
            d_map_sorted = AvaDeviceArray<uint32_t, int>::create({(int) n});

            d_children = AvaDeviceArray<uint32_t, int>::create({(int) (n-1), 2});
            d_range = AvaDeviceArray<uint32_t, int>::create({(int) (n-1), 2});
            d_delta = AvaDeviceArray<uint64_t, int>::create({(int) (n-1)});
            d_split_idx = AvaDeviceArray<uint32_t, int>::create({(int) (n-1)});

            d_internal_data = AvaDeviceArray<DataT, int>::create({(int) (2*n-1)});
            d_obj_m = AvaDeviceArray<ObjT, int>::create({(int) n});
            d_obj = objects; // copy pointer

            d_touched = AvaDeviceArray<uint32_t, int>::create({(int) (n-1)});
            d_parent = AvaDeviceArray<uint32_t, int>::create({(int) (2*n-1)});
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
            uint32_t const n_v = n;
            ava_for<AVA_BLOCK_SIZE>(0, 0, n_v, [=] __device__ (size_t const tid) {
                d_obj_m_v(tid) = d_obj_v(d_map_sorted_v(tid));

                // At the same time, init the leaf nodes
                DataF functor;
                ObjT p = d_obj_m_v(tid);
                DataComputeT data = functor.init(p);
                d_internal_data_v(tid) = data;

                // And compute the deltas
                if (tid < n_v-1) d_delta_v(tid) = d_morton_sorted_v(tid) ^ d_morton_sorted_v(tid+1);

                // And reset the touched flag 
                d_touched_v(tid) = 0;
            });
        }

        // Build the hierachy of the LBVH
        void _build_hierarchy() {
            AvaView<uint64_t, -1> d_delta_v         = d_delta->to_view<-1>();
            AvaView<uint32_t, -1> d_split_idx_v     = d_split_idx->to_view<-1>();
            AvaView<DataT, -1>    d_internal_data_v = d_internal_data->template to_view<-1>();
            AvaView<uint32_t, -1> d_touched_v       = d_touched->to_view<-1>();
            AvaView<uint32_t, -1> d_parent_v        = d_parent->to_view<-1>();
            AvaView<uint32_t, -1, 2> d_range_v     = d_range->to_view<-1, 2>();
            AvaView<uint32_t, -1, 2> d_children_v    = d_children->to_view<-1, 2>();
            AvaView<uint32_t, -1> d_root_v          = d_root->to_view<-1>();
            uint32_t n_v = static_cast<uint32_t>(n);

            ava_for<AVA_BLOCK_SIZE>(0, 0, n_v, [=] __device__ (uint32_t const i) {
                uint32_t idx = i;

                DataF functor;
                DataT data = d_internal_data_v(i);
                uint32_t imin = i;
                uint32_t imax = i;
                uint32_t parent;
                while (true) {
                    // We are at the root
                    if (imin == 0 && imax == n_v-1) {
                        d_parent_v(idx) = 2*n_v;
                        d_root_v(0) = idx;
                        break;
                    }

                    // Choose whether we attach this node to the right or left parent
                    bool const right_parent = imin == 0 || (imax != n_v-1 && d_delta_v(imax) < d_delta_v(imin-1));

                    parent = right_parent ? imax+n_v : imin+n_v-1;
                    d_range_v(parent-n_v, !right_parent) = right_parent ? imin : imax;
                    d_children_v(parent-n_v, !right_parent) = idx;
                    d_parent_v(idx) = parent;
                    __threadfence();

                    if (ava::atomic::fetch_add(&d_touched_v(parent-n_v), 1U)) {
                        uint32_t const sibling = d_children_v(parent - n_v, right_parent);

                        DataT sibling_data = d_internal_data_v(sibling);
                        functor.combine(data, sibling_data);
                        d_internal_data_v(parent) = data;

                        d_split_idx_v(parent-n_v) = right_parent ? imax : (imin-1);

                        idx = parent;
                        imin = d_range_v(parent-n_v, 0);
                        imax = d_range_v(parent-n_v, 1);
                    } else {
                        // Terminate thread
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
        }

    };

} // namespace stream::geo

#endif // __STREAM_LBVH_AGGLO_HPP__

