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
#ifndef __STREAM_VEC_HPP__
#define __STREAM_VEC_HPP__

#include "defines.h"
#include <initializer_list>
#include <stdint.h>
#include <utility>
#include <cmath>

/* file stream/include/geometry/vec.hpp
 *
 * Define the templated structure : 
 *      Vec<typename Scalar, int dim> 
 *
 * For use with 2, 3 and 4 dimension tuples 
 */

namespace stream::geo {
    // Forward declaration
    template<typename Scalar, int dim> struct BBox;
    template<typename Scalar, int dim> struct Vec; 
}

// Typedefs for often used template arguments 
#ifdef __cplusplus
extern "C" {
#endif
    typedef stream::geo::Vec<float, 2> Vec2f;
    typedef stream::geo::Vec<float, 3> Vec3f;
    typedef stream::geo::Vec<float, 4> Vec4f;

    typedef stream::geo::Vec<double, 2> Vec2d;
    typedef stream::geo::Vec<double, 3> Vec3d;
    typedef stream::geo::Vec<double, 4> Vec4d;

    typedef stream::geo::Vec<fp_tt, 2> Vec2fp;
    typedef stream::geo::Vec<fp_tt, 3> Vec3fp;
    typedef stream::geo::Vec<fp_tt, 4> Vec4fp;
#ifdef __cplusplus
}
#endif

#ifdef __CUDACC__
#include <cuda/std/limits>
template<typename T>
using limits = cuda::std::numeric_limits<T>;
#else 
#include <limits>
template<typename T>
using limits = std::numeric_limits<T>;
#endif

namespace stream::geo {

template<typename _Scalar, int dim>
struct Vec {
    using VecT = Vec<_Scalar, dim>;
    using BBoxT = BBox<_Scalar, dim>;
    using Scalar = _Scalar;

    Scalar coord[dim];

    Vec() noexcept = default;
    __host__ __device__ constexpr Vec(std::initializer_list<Scalar> init) noexcept {
        using size_type = typename std::initializer_list<Scalar>::size_type;
        auto iter = init.begin();
        for (size_type i = 0; i < init.size(); i++) {
            coord[i] = *iter;
            iter++;
        }
    }

    // ========================= Copy / move constructors =====================
    Vec(const VecT& other) noexcept = default;
    Vec(VecT&& other) noexcept = default;

    // ========================= Copy / move assignments ======================
    __host__ __device__ VecT& operator=(const VecT& other) noexcept {
        for (int i = 0; i < dim; ++i){
            coord[i] = other.coord[i];
        }
        return *this;
    }

    __host__ __device__ VecT& operator=(const VecT&& other) noexcept {
        for (int i = 0; i < dim; ++i){
            coord[i] = std::move(other.coord[i]);
        }
        return *this;
    }

    // ================== In place addition/subtraction ==================
    __host__ __device__ VecT& operator+=(const VecT& other) noexcept {
        for (int i = 0; i < dim; ++i) {
            coord[i] += other.coord[i];
        }
        return *this;
    }

    __host__ __device__ VecT& operator+=(const Scalar other) noexcept {
        for (int i = 0; i < dim; ++i) {
            coord[i] += other;
        }
        return *this;
    }

    __host__ __device__ VecT& operator-=(const VecT& other) noexcept {
        for (int i = 0; i < dim; ++i) {
            coord[i] -= other.coord[i];
        }
        return *this;
    }

    __host__ __device__ VecT& operator-=(const Scalar other) noexcept {
        for (int i = 0; i < dim; ++i) {
            coord[i] -= other;
        }
        return *this;
    }

    // ================== In place multiplication by scalar ==============
    __host__ __device__ VecT& operator*=(Scalar const s) noexcept {
        for (int i = 0; i < dim; ++i) {
            coord[i] *= s;
        }
        return *this;
    }

    // ================== Out of place binary operators ==================
    __host__ __device__ friend VecT operator+(VecT lhs, const VecT& rhs) noexcept {
        lhs += rhs;
        return lhs;
    };
    __host__ __device__ friend VecT operator+(VecT lhs, const Scalar rhs) noexcept {
        lhs += rhs;
        return lhs;
    };
    __host__ __device__ friend VecT operator+(const Scalar lhs, VecT rhs) noexcept {
        rhs += lhs;
        return rhs;
    };

    __host__ __device__ friend VecT operator-(VecT lhs, const VecT& rhs) noexcept {
        lhs -= rhs;
        return lhs;
    };
    __host__ __device__ friend VecT operator-(VecT lhs, const Scalar rhs) noexcept {
        lhs -= rhs;
        return lhs;
    };
    __host__ __device__ friend VecT operator-(const Scalar lhs, VecT rhs) noexcept {
        return -rhs + lhs;
    };

    __host__ __device__ friend VecT operator*(VecT lhs, const Scalar rhs) noexcept {
        lhs *= rhs;
        return lhs;
    };
    __host__ __device__ friend VecT operator*(const Scalar lhs, VecT rhs) noexcept {
        rhs *= lhs;
        return rhs;
    };

    __host__ __device__ friend VecT operator/(VecT lhs, const Scalar rhs) noexcept {
        Scalar const inv = static_cast<Scalar>(1)/rhs;
        lhs *= inv;
        return lhs;
    };

    // Component wise multiplication
    __host__ __device__ friend VecT cMult(const VecT& p, const VecT& q) noexcept {
        VecT ret;
        for (int i = 0; i < dim; ++i) {
            ret[i] = p[i]*q[i];
        }
        return ret;
    }

    // ================= Unary - ==========================================
    __host__ __device__ friend VecT operator-(const VecT& p) noexcept {
        VecT ret;
        for (int i = 0; i < dim; ++i) {
            ret[i] = -p[i];
        }
        return ret;
    };

    // ================ Accessor ==========================================
    __host__ __device__ constexpr Scalar& operator[](int i)       noexcept {return coord[i];};
    __host__ __device__ constexpr Scalar  operator[](int i) const noexcept {return coord[i];};

    // ================== Geometric / linear algebra functions ============
    __host__ __device__ Scalar norm() const noexcept {
        return std::sqrt(sqnorm());
    }

    __host__ __device__ Scalar sqnorm() const noexcept {
        Scalar sum = static_cast<Scalar>(0);
        for (int i = 0; i < dim; ++i) {
            sum += coord[i] * coord[i];
        }
        return sum;
    }

    __host__ __device__ Scalar dot(const VecT& other) const noexcept {
        Scalar sum = static_cast<Scalar>(0);
        for (int i = 0; i < dim; ++i) {
            sum += coord[i] * other[i];
        }
        return sum;
    }

    __host__ __device__ VecT unit() const noexcept {
        return (*this) / norm();
    }

    __host__ __device__ Scalar cross(const VecT& other) const noexcept requires (dim == 2) {
        return coord[0]*other.coord[1] - coord[1]*other.coord[0];
    }
    __host__ __device__ VecT cross(const VecT& other) const noexcept requires (dim == 3) {
        return Vec<Scalar, dim>({
            (coord[1]*other.coord[2] - coord[2]*other.coord[1]),
            (coord[2]*other.coord[0] - coord[0]*other.coord[2]),   
            (coord[0]*other.coord[1] - coord[1]*other.coord[0])
        });
    }

    __host__ __device__ static constexpr VecT minPoint() noexcept {
        VecT p;
        for (int i = 0; i < dim; ++i) {
            p[i] = - limits<Scalar>::max();
        }
        return p;
    }

    __host__ __device__ static constexpr VecT maxPoint() noexcept {
        VecT p;
        for (int i = 0; i < dim; ++i) {
            p[i] = limits<Scalar>::max();
        }
        return p;
    };

    // To be able to use them in the quadtree 
    __host__ __device__ VecT get_centroid() const noexcept {
        return *this;
    }

    __host__ __device__ BBox<Scalar, dim> get_bbox() const noexcept {
        return BBox<Scalar, dim>(*this, *this);
    }
};

} // namespace stream::geo

#endif // __STREAM_VEC_HPP__
