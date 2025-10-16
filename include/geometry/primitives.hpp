#ifndef __STREAM_PRIMITIVES_HPP__
#define __STREAM_PRIMITIVES_HPP__

#include "defines.h"
#include "bbox.hpp"
#include "vec.hpp"

// File stream/geometry/primitives.hpp 
//
// Defines a set of raw geometric primitives and their indexed 
// equivalent (suffixed by "Elem").
//
// - Raw geometric primitives store the coordinates directly. 
//   E.g. an edge from (0, 0) to (1, 1) is stored as a pair [(0, 0), (1, 1)].
// - Indexed geometric primitives are stored through indices in a global 
//   coordinate array. 
//   E.g. an edge from (0, 0) to (1, 1) is stored as a pair (4, 10) where 
//        4 is the index of (0, 0) and 10 is the index of (1, 1) in the global 
//        coordinate array.
//
// Set of implemented primitives :
//   Sphere : define a N-dimensional sphere with center `c` and radius `r`
//   SphereElem : define an indexed N-dimensional sphere with radius `r` and 
//                center taken at index `c` in a given coordinate array.
//  
//   Edge : define an N-dimensional edge with source point `s` and target point 
//          `t`.
//   EdgeElem : define an indexed N-dimensional edge with source and target points 
//              taken at indices `s` and `t` in a given coordinate array.
//  
//   Tri : define a N-dimensional triangle with vertices `a`, `b`, `c`. 
//         The order of the vertices changes the orientation.
//   TriElem : define an indexed N-dimensional triangle with vertices taken at 
//             indices `a`, `b`, `c` of a given coordinate array. The order of the 
//             vertices changes the orientation.
//
//   Tet : define a N-dimensional tetrahedron with vertices `a`, `b`, `c`, `d`. 
//         The order of the vertices changes the orientation.
//   TetElem : define an indexed N-dimensional tetrahedron with vertices taken at 
//             indices `a`, `b`, `c`, `d` of a given coordinate array. The order of the 
//             vertices changes the orientation.
//
// All raw primitives have two methods :
// - `get_centroid()` : returns the centroid of the primitive 
// - `get_bbox()` : return the bounding box of the primitive
// And use two template parameters :
// - _Scalar : type of scalar coordinate used by the primitive (float, double)
// - dim : dimension of the space (2, 3)
//
// All indexed primitives have two methods :
// - `get_centroid(CoordArray c)` : returns the centroid of the primitive by 
//                                  using the global coordinate array `c`
// - `get_bbox(CoordArray c)` : returns the bounding box of the primitive by 
//                              using the global coordinate array `c`
//
// And use three template parameters :
// - _Scalar : type of scalar coordinate used by the primitive (float, double)
// - dim : dimension of the space (2, 3)
// - _IdxType : type of the indices used to retrieve coordinates (uint32, int, long...)
// The CoordArray type must implement `operator[](IdxType i)` to return the
// coordinate at index i. A raw pointer is valid.

namespace stream::geo {
    // Forward declarations
    template <typename _Scalar, int dim> struct Sphere;
    template <typename _Scalar, int dim> struct Edge;
    template <typename _Scalar, int dim> struct Tri;
    template <typename _Scalar, int dim> struct Tet;

    template <typename _Scalar, int dim, typename _IdxType = int> struct SphereElem;
    template <typename _Scalar, int dim, typename _IdxType = int> struct EdgeElem;
    template <typename _Scalar, int dim, typename _IdxType = int> struct TriElem;
    template <typename _Scalar, int dim, typename _IdxType = int> struct TetElem;
}  // namespace stream::geo

#ifdef __cplusplus
extern "C" {
#endif
    // Typedefs
    typedef stream::geo::Sphere<fp_tt, 2> Sphere2D;
    typedef stream::geo::Sphere<fp_tt, 3> Sphere3D;
    typedef stream::geo::Edge<fp_tt, 2> Edge2D;
    typedef stream::geo::Edge<fp_tt, 3> Edge3D;
    typedef stream::geo::Tri<fp_tt, 2> Tri2D;
    typedef stream::geo::Tri<fp_tt, 3> Tri3D;
    typedef stream::geo::Tet<fp_tt, 3> Tet3D; // Cannot have Tet in 2D
                                 
    typedef stream::geo::SphereElem<fp_tt, 2> SphereElem2D;
    typedef stream::geo::SphereElem<fp_tt, 3> SphereElem3D;
    typedef stream::geo::EdgeElem<fp_tt, 2> EdgeElem2D;
    typedef stream::geo::EdgeElem<fp_tt, 3> EdgeElem3D;
    typedef stream::geo::TriElem<fp_tt, 2> TriElem2D;
    typedef stream::geo::TriElem<fp_tt, 3> TriElem3D;
    typedef stream::geo::TetElem<fp_tt, 3> TetElem3D; // Cannot have Tet in 2D
    
    typedef stream::geo::BBox<fp_tt, 2> BBox2D;
    typedef stream::geo::BBox<fp_tt, 3> BBox3D;
#ifdef __cplusplus
}
#endif
    
// =========================== Declarations ===================================
namespace stream::geo {
    template <typename _Scalar, int dim>
    struct Sphere {
        using Scalar = _Scalar; 
        using BBoxT  = BBox<Scalar, dim>;
        using VecT   = Vec<Scalar, dim>;

        VecT   c;  // center
        Scalar r;  // radius
        
        VecT __host__ __device__ inline get_centroid() const noexcept {
            return c;
        }

        BBoxT __host__ __device__ inline get_bbox() const noexcept {
            BBoxT ret = c.get_bbox();
            ret.pmin -= r;
            ret.pmax += r;
            return ret;
        }
    };

    template <typename _Scalar, int dim, typename _IdxType>
    struct SphereElem {
        using Scalar  = _Scalar; 
        using IdxType = _IdxType;
        using BBoxT   = BBox<Scalar, dim>;
        using VecT    = Vec<Scalar, dim>;

        IdxType c;
        Scalar r;  // radius
        
        template<typename CoordArray>
        VecT __host__ __device__ inline get_centroid(const CoordArray& coords) const noexcept {
            return Sphere<Scalar, dim>(coords[c], r).get_centroid();
        }

        template<typename CoordArray>
        BBoxT __host__ __device__ inline get_bbox(const CoordArray& coords) const noexcept {
            return Sphere<Scalar, dim>(coords[c], r).get_bbox();
        }
    };


    template <typename _Scalar, int dim>
    struct Edge {
        using Scalar = _Scalar; 
        using BBoxT  = BBox<Scalar, dim>;
        using VecT   = Vec<Scalar, dim>;

        VecT s; // source
        VecT t; // target
        
        VecT __host__ __device__ inline get_centroid() const noexcept {
            return static_cast<Scalar>(0.5) * (s + t);
        }

        BBoxT __host__ __device__ inline get_bbox() const noexcept {
            BBoxT ret;
            for (int i = 0; i < dim; i++){
                ret.min(i) = std::fmin(s[i], t[i]);
                ret.max(i) = std::fmax(s[i], t[i]);
            }
            return ret;
        };
    };


    template <typename _Scalar, int dim, typename _IdxType>
    struct EdgeElem {
        using Scalar = _Scalar; 
        using IdxType = _IdxType;
        using BBoxT  = BBox<Scalar, dim>;
        using VecT   = Vec<Scalar, dim>;

        IdxType s; // source
        IdxType t; // target
        
        template<typename CoordArray>
        VecT __host__ __device__ inline get_centroid(const CoordArray& coords) const noexcept {
            return Edge<Scalar, dim>(coords[s], coords[t]).get_centroid();
        }

        template<typename CoordArray>
        BBoxT __host__ __device__ inline get_bbox(const CoordArray& coords) const noexcept {
            return Edge<Scalar, dim>(coords[s], coords[t]).get_bbox();
        };
    };


    template <typename _Scalar, int dim>
    struct Tri {
        using Scalar = _Scalar; 
        using BBoxT  = BBox<Scalar, dim>;
        using VecT   = Vec<Scalar, dim>;

        VecT a;
        VecT b;
        VecT c;

        VecT __host__ __device__ inline get_centroid() const{
            constexpr const Scalar one_third = static_cast<Scalar>(1./3.);
            return one_third * (a + b + c);
        }

        BBoxT __host__ __device__ inline get_bbox() const {
            BBoxT ret;
            for (int i = 0; i < dim; i++){
                ret.min(i) = std::fmin(a[i], std::fmin(b[i], c[i]));
                ret.max(i) = std::fmax(a[i], std::fmax(b[i], c[i]));
            }
            return ret;
        };
    };

    template <typename _Scalar, int dim, typename _IdxType>
    struct TriElem {
        using Scalar  = _Scalar; 
        using IdxType = _IdxType;
        using BBoxT   = BBox<Scalar, dim>;
        using VecT    = Vec<Scalar, dim>;

        IdxType a;
        IdxType b;
        IdxType c;

        template<typename CoordArray>
        VecT __host__ __device__ inline get_centroid(const CoordArray& coords) const noexcept {
            return Tri<Scalar, dim>(coords[a], coords[b], coords[c]).get_centroid();
        };

        template<typename CoordArray>
        BBoxT __host__ __device__ inline get_bbox(const CoordArray& coords) const noexcept {
            return Tri<Scalar, dim>(coords[a], coords[b], coords[c]).get_bbox();
        };
    };
    
    template <typename _Scalar, int dim>
    struct Tet {
        using Scalar = _Scalar; 
        using BBoxT  = BBox<Scalar, dim>;
        using VecT   = Vec<Scalar, dim>;

        VecT a;
        VecT b;
        VecT c;
        VecT d;

        VecT __host__ __device__ inline get_centroid() const{
            return static_cast<Scalar>(0.25) * (a + b + c + d);
        }

        BBoxT __host__ __device__ inline get_bbox() const {
            BBoxT ret;
            for (int i = 0; i < dim; i++){
                ret.min(i) = std::fmin(std::fmin(a[i], b[i]), std::fmin(c[i], d[i]));
                ret.max(i) = std::fmax(std::fmax(a[i], b[i]), std::fmax(c[i], d[i]));
            }
            return ret;
        };
    };

    template <typename _Scalar, int dim, typename _IdxType>
    struct TetElem {
        using Scalar  = _Scalar; 
        using IdxType = _IdxType;
        using BBoxT   = BBox<Scalar, dim>;
        using VecT    = Vec<Scalar, dim>;

        IdxType a;
        IdxType b;
        IdxType c;
        IdxType d;

        template<typename CoordArray>
        VecT __host__ __device__ inline get_centroid(const CoordArray& coords) const noexcept {
            return Tet<Scalar, dim>(coords[a], coords[b], coords[c], coords[d]).get_centroid();
        };

        template<typename CoordArray>
        BBoxT __host__ __device__ inline get_bbox(const CoordArray& coords) const noexcept {
            return Tet<Scalar, dim>(coords[a], coords[b], coords[c], coords[d]).get_bbox();
        };
    };

    template <typename Primitive>
    struct ComputeBB_Functor {
        using BBoxT = Primitive::BBoxT;

        BBoxT __host__ __device__ init(const Primitive& prim) const noexcept {
            return prim.get_bbox();
        }

        void __host__ __device__ combine(BBoxT& lhs, const BBoxT& rhs) const noexcept {
            lhs.combineBox(rhs);
        }

        BBoxT __host__ __device__ finalize(const BBoxT& data) const noexcept {
            return data;
        }
    };

    template <typename Primitive, typename CoordArray>
    struct ComputeElemBB_Functor {
        using BBoxT = Primitive::BBoxT;
        const CoordArray& coords;

        BBoxT __host__ __device__ init(const Primitive& prim) const noexcept {
            return prim.get_bbox(coords);
        }

        void __host__ __device__ combine(BBoxT& lhs, const BBoxT& rhs) const noexcept {
            lhs.combineBox(rhs);
        }

        BBoxT __host__ __device__ finalize(const BBoxT& data) const noexcept {
            return data;
        }
    };

}


#endif // __STREAM_PRIMITIVES_HPP__
