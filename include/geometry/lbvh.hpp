#ifndef __STREAM_LBVH_HPP__
#define __STREAM_LBVH_HPP__

/*
 *  Implementation of the Linear Bounding Volume Hierarchy (LBVH) from :
 *     T. Karras 2012, Eurographics. 
 *     "Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees"
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
    template<typename ObjT, typename DataF, int dim> struct LBVH;
} // namespace stream::geo

// Typedefs for often used template arguments
#ifdef __cplusplus
extern "C" {
#endif

    typedef stream::geo::LBVH<Vec2f, stream::geo::ComputeBB_Functor<Vec2f>, 2> Point2DLBVH;
    typedef stream::geo::LBVH<Sphere2D, stream::geo::ComputeBB_Functor<Sphere2D>, 2> Sphere2DLBVH;
    typedef stream::geo::LBVH<Edge2D, stream::geo::ComputeBB_Functor<Edge2D>, 2> Edge2DLBVH;
    typedef stream::geo::LBVH<Tri2D, stream::geo::ComputeBB_Functor<Tri2D>, 2> Tri2DLBVH;
    
    typedef stream::geo::LBVH<Sphere3D, stream::geo::ComputeBB_Functor<Sphere3D>, 3> Sphere3DLBVH;
#ifdef __cplusplus
}
#endif

namespace stream::geo {

    template<typename ObjT, typename DataF, int dim>
    struct LBVH {
        using Scalar = typename ObjT::Scalar;
        using DataComputeT = decltype(std::declval<DataF>().init(ObjT()));
        using DataT = decltype(std::declval<DataF>().finalize(DataComputeT()));
        using VecT = Vec<Scalar, dim>;
        using BBoxT = BBox<Scalar, dim>;

        uint32_t n; // number of objects
                    
        size_t tmp_size; // Size of temporary storage in bytes
        AvaDeviceArray<char, size_t>::Ptr tmp; // Temporary storage
                                 
        typename AvaDeviceArray<BBoxT, int>::Ptr d_bb_glob; // BBox of all objects

        AvaDeviceArray<uint64_t, int>::Ptr d_morton;     // d_morton[i] = morton(d_coords[i])
        AvaDeviceArray<uint64_t, int>::Ptr d_morton_sorted; // Sorted morton codes

        AvaDeviceArray<uint32_t, int>::Ptr d_map;         // 0, 1, 2, 3, ..., n-1
        AvaDeviceArray<uint32_t, int>::Ptr d_map_sorted;  // d_map sorted accorting to the morton codes

        AvaDeviceArray<int, int>::Ptr     d_internal_sep;  // Internal node separations
        AvaDeviceArray<uint8_t, int>::Ptr d_child_is_leaf; // if left or right child is a leaf of internal node i

        typename AvaDeviceArray<DataT, int>::Ptr d_internal_data;  // Data stored on each internal node
                                  
        typename AvaDeviceArray<ObjT, int>::Ptr d_obj;   // Input objects : store only the pointer.
        typename AvaDeviceArray<ObjT, int>::Ptr d_obj_m; // objects sorted according to morton codes
                                   
        AvaDeviceArray<uint32_t, int>::Ptr d_touched; // Has the node already been touched (for the data bottom-up parallel compute)
        AvaDeviceArray<uint32_t, int>::Ptr d_leaf_parent; // The parent of the leaves (L in the paper)
        AvaDeviceArray<uint32_t, int>::Ptr d_internal_parent; // The parent of the internal nodes (I in the paper)

        LBVH() {
            n = 0;
            tmp_size = 0;

            tmp = AvaDeviceArray<char, size_t>::create({0});
            d_bb_glob = AvaDeviceArray<BBoxT, int>::create({1});

            d_morton = AvaDeviceArray<uint64_t, int>::create({0});
            d_morton_sorted = AvaDeviceArray<uint64_t, int>::create({0});

            d_map = AvaDeviceArray<uint32_t, int>::create({0});
            d_map_sorted = AvaDeviceArray<uint32_t, int>::create({0});

            d_internal_sep = AvaDeviceArray<int, int>::create({0});
            d_child_is_leaf = AvaDeviceArray<uint8_t, int>::create({0});

            d_internal_data = AvaDeviceArray<DataT, int>::create({0});
            d_obj_m = AvaDeviceArray<ObjT, int>::create({0});
            // d_obj = objects; // copy pointer

            d_touched = AvaDeviceArray<uint32_t, int>::create({0});
            d_leaf_parent = AvaDeviceArray<uint32_t, int>::create({0});
            d_internal_parent = AvaDeviceArray<uint32_t, int>::create({0});
        }

        void set_objects(typename AvaDeviceArray<ObjT, int>::Ptr objects) {
            n = objects->size;

            d_morton->resize({(int) n});
            d_morton_sorted->resize({(int) n});

            d_map->resize({(int) n});
            d_map_sorted->resize({(int) n});

            d_internal_sep->resize({(int) (n-1)});
            d_child_is_leaf->resize({(int) (n-1)});

            d_internal_data->resize({(int) (n-1)});
            d_obj_m->resize({(int) n});
            d_obj = objects; // copy pointer

            d_touched->resize({(int) (n-1)});
            d_leaf_parent->resize({(int) n});
            d_internal_parent->resize({(int) (n-1)});
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
                constexpr uint64_t imax = 1ULL << (64 / dim);
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
            ava_for<AVA_BLOCK_SIZE>(0, 0, n, [=] __device__ (size_t const tid) {
                d_obj_m_v(tid) = d_obj_v(d_map_sorted_v(tid));
            });
        }

        // Build the hierachy of the LBVH
        void _build_hierarchy() {
            AvaView<uint32_t, -1> d_touched_v         = d_touched->to_view<-1>();
            AvaView<uint64_t, -1> d_morton_sorted_v   = d_morton_sorted->to_view<-1>();
            AvaView<int, -1>      d_internal_sep_v    = d_internal_sep->to_view<-1>();
            AvaView<uint32_t, -1> d_leaf_parent_v     = d_leaf_parent->to_view<-1>();
            AvaView<uint32_t, -1> d_internal_parent_v = d_internal_parent->to_view<-1>();
            AvaView<uint8_t, -1>  d_child_is_leaf_v   = d_child_is_leaf->to_view<-1>();
            int n_v = n;
            ava_for<AVA_BLOCK_SIZE>(0, 0, n-1, [=] __device__ (int const i) {
                d_touched_v(i) = 0;
                uint64_t const mi = d_morton_sorted_v(i);
                int const deltap1 = (i+1 >= n_v) ? -1 : __builtin_clzll(mi ^ d_morton_sorted_v(i+1));
                int const deltam1 = (i-1 < 0) ? -1 : __builtin_clzll(mi ^ d_morton_sorted_v(i-1));
                int const d = (deltap1 - deltam1 > 0) ? 1 : -1;
                int const dmin = (i-d < n_v && i-d >= 0) ? __builtin_clzll(mi ^ d_morton_sorted_v(i-d)) : -1; 

                int lmax = 2;
                while (i + lmax*d >= 0 && i + lmax*d < n_v && __builtin_clzll(mi ^ d_morton_sorted_v(i + (lmax * d))) > dmin) {
                    lmax *= 2;
                }
                int l = 0;
                for (int t = lmax / 2; t >= 1; t /= 2) {
                    if (i + (l+t)*d >= 0 && i + (l+t)*d < n_v && __builtin_clzll(mi ^ d_morton_sorted_v(i + (l + t) * d)) > dmin) {
                        l += t;
                    }
                }
                int const j = i + l * d;
                int gamma   = (i + j) >> 1;
                uint64_t const mj = d_morton_sorted_v(j);
                if (mi != mj) {
                    int dnode = __builtin_clzll(mi ^ mj);
                    int s     = 0;
                    for (int t = (l + 1) / 2; t >= 1; t = (t + 1) / 2) {
                        if (i+(s+t)*d >= 0 && i+(s+t)*d < n_v && __builtin_clzll(mi ^ d_morton_sorted_v(i + (s + t) * d)) > dnode) {
                            s += t;
                        }
                        if (t == 1) break;
                    }
                    gamma = i + s * d + (d < 0 ? d : 0);
                }

                d_internal_sep_v(i) = gamma;
                uint8_t child_is_leaf_loc = 0;
                if (gamma == (i < j ? i : j)) {
                    child_is_leaf_loc   |= 0b00000001; // 1
                    d_leaf_parent_v(gamma) = i;
                } else {
                    child_is_leaf_loc       &= 0b11111110; // ~1
                    d_internal_parent_v(gamma) = i;
                }
                if (gamma + 1 == (i > j ? i : j)) {
                    child_is_leaf_loc       |= 0b00000010; // 2
                    d_leaf_parent_v(gamma + 1) = i | 0x80000000;
                } else {
                    child_is_leaf_loc           &= 0b11111101; // ~2
                    d_internal_parent_v(gamma + 1) = i | 0x80000000;
                }
                if (i == 0) {
                    d_internal_parent_v(i) = n_v;
                }
                d_child_is_leaf_v(i) = child_is_leaf_loc;
            });
        }

        // Fit internal node data
        void fit(){
            AvaView<DataT, -1> d_internal_data_v = d_internal_data->template to_view<-1>();
            AvaView<ObjT, -1> d_obj_m_v = d_obj_m->template to_view<-1>();
            AvaView<uint32_t, -1> d_touched_v = d_touched->to_view<-1>();
            AvaView<int, -1> d_internal_sep_v = d_internal_sep->to_view<-1>();
            AvaView<uint32_t, -1> d_leaf_parent_v = d_leaf_parent->to_view<-1>();
            AvaView<uint32_t, -1> d_internal_parent_v = d_internal_parent->to_view<-1>();
            AvaView<uint8_t, -1> d_child_is_leaf_v = d_child_is_leaf->to_view<-1>();
            uint32_t n_v = static_cast<uint32_t>(n);

            ava_for<AVA_BLOCK_SIZE>(0, 0, n, [=] __device__ (size_t const i) {
                DataF functor;
                ObjT p = d_obj_m_v(i);
                DataComputeT data = functor.init(p);
                uint32_t parent = d_leaf_parent_v(i);
                bool isRight = (parent & 0x80000000) != 0;
                parent = parent & 0x7FFFFFFF;

                while (parent < n_v && ava::atomic::fetch_or(&d_touched_v(parent), 1U)) {
                    uint32_t child_id = d_internal_sep_v(parent);
                    uint8_t child_is_leaf = d_child_is_leaf_v(parent);
                    if (child_is_leaf & (!isRight + 1)) { // Check if the other node is leaf
                        p = d_obj_m_v(child_id+!isRight);
                        DataComputeT other_data = functor.init(p);
                        functor.combine(data, other_data);
                    } else {
                        DataComputeT const vdata = d_internal_data_v(child_id+!isRight);
                        functor.combine(data, vdata);
                    }

                    d_internal_data_v(parent) = functor.finalize(data);
                    __threadfence(); // WARNING : this statement makes sure vdata 
                                     // above is written by this thread in 
                                     // d_internal_data BEFORE it is read by another

                    parent = d_internal_parent_v(parent);
                    isRight = (parent & 0x80000000) != 0;
                    parent = parent & 0x7FFFFFFF;
                }
            });
        }

        // Build the LBVH tree
        void build(){
            _compute_global_bbox();
            _compute_morton_codes();
            _sort_objects();
            _build_hierarchy();
            fit();
        }

    };

} // namespace stream::geo

#endif // __STREAM_LBVH_HPP__

