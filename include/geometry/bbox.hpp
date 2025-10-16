#ifndef __GENERIC_LBVH_BBOX_H__
#define __GENERIC_LBVH_BBOX_H__

#include <utility>
#include "defines.h"
#include "vec.hpp"

// Forward declaration
namespace stream::geo {
    template<typename Scalar, int dim> struct BBox;
    template<typename Scalar, int dim> struct Vec; 
}

#ifdef __cplusplus
extern "C" {
#endif

    typedef stream::geo::BBox<float, 2>  BBox2f;
    typedef stream::geo::BBox<float, 3>  BBox3f;
    
    typedef stream::geo::BBox<double, 2> BBox2d;
    typedef stream::geo::BBox<double, 3> BBox3d;

#ifdef __cplusplus
}
#endif

namespace stream::geo {
template<typename Scalar, int dim>
struct BBox {
    using VecT = Vec<Scalar, dim>;
    using BBoxT = BBox<Scalar, dim>;

    VecT pmin;
    VecT pmax;

    // Default : non-initialized memory
    BBox() noexcept = default;
    BBox(const BBoxT& other) noexcept = default;
    BBox(BBoxT&& other) noexcept = default;

    // Init a bounding box using the bottom-left and top-right corners
    __host__ __device__ constexpr BBox(const VecT& _pmin, const VecT& _pmax) noexcept : pmin(_pmin), pmax(_pmax) {};

    __host__ __device__ BBox& operator=(const BBoxT& other) noexcept {
        pmin = other.pmin;
        pmax = other.pmax;
        return *this;
    }

    __host__ __device__ BBox& operator=(BBoxT&& other) noexcept {
        pmin = std::move(other.pmin);
        pmax = std::move(other.pmax);
        return *this;
    }

    // ============================ Accessors =================================
    // Read-Write
    __host__ __device__ Scalar& min(int i) noexcept { return pmin[i]; }
    __host__ __device__ Scalar& max(int i) noexcept { return pmax[i]; }

    // Read-Only
    __host__ __device__ Scalar min(int i) const noexcept { return pmin[i]; }
    __host__ __device__ Scalar max(int i) const noexcept { return pmax[i]; }


    // ========================= Enlarge bounding box =========================
    // Combine the bounding box with another in-place (this bbox is modified)
    __host__ __device__ void combineBox(const BBoxT& other) noexcept {
        for (int i = 0; i < dim; i++) {
            pmin[i] = std::fmin(pmin[i], other.pmin[i]);
        }

        for (int i = 0; i < dim; i++) {
            pmax[i] = std::fmax(pmax[i], other.pmax[i]);
        }
    }

    // Add a point to the bounding box in-place (this bbox is modified)
    __host__ __device__ void combinePoint(const VecT& other) noexcept {
        for (int i = 0; i < dim; i++) {
            pmin[i] = std::fmin(pmin[i], other[i]);
        }
        for (int i = 0; i < dim; i++) {
            pmax[i] = std::fmax(pmax[i], other[i]);
        }
    }

    // =================== Requirements for Quadtree indexing =================
    __host__ __device__ VecT get_centroid() const noexcept {
        return static_cast<Scalar>(0.5) * (pmin + pmax);
    }
    __host__ __device__ BBoxT get_bbox() const noexcept {
        return *this;
    }

    // Get an empty bounding box
    __host__ __device__ constexpr static BBoxT empty() noexcept {
        constexpr BBoxT ret = BBoxT(VecT::maxPoint(), VecT::minPoint());
        return ret;
    }
};

} // namespace stream::geo

#endif // __GENERIC_LBVH_BBOX_H__
