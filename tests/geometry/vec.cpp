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
#include <cassert>
#include <stdio.h>
#include "geometry/vec.hpp"

#ifdef __CUDACC__

__global__ void kernel_test() {
    Vec2f a({1.0f, 2.0f});
    Vec2f b({3.0f, 4.0f});

    Vec2f test_vec_add = a + b;
    if (!(test_vec_add[0] == 4.0f && test_vec_add[1] == 6.0f)) {
        printf("[GPU] vec + vec failed\n");
    }

    Vec2f test_scal_add_left = 1.0f + a;
    if (!(test_scal_add_left[0] == 2.0f && test_scal_add_left[1] == 3.0f)) {
        printf("[GPU] scalar + vec failed\n");
    }

    Vec2f test_scal_add_right = a + 1.0f;
    if (!(test_scal_add_right[0] == 2.0f && test_scal_add_right[1] == 3.0f)) {
        printf("[GPU] vec + scalar failed\n");
    }

    Vec2f test_vec_sub = a - b;
    if (!(test_vec_sub[0] == -2.0f && test_vec_sub[1] == -2.0f)){
        printf("[GPU] vec - vec failed\n");
    }

    Vec2f test_scal_mult_right = a * 2.0f;
    if (!(test_scal_mult_right[0] == 2.0f && test_scal_mult_right[1] == 4.0f)){
        printf("[GPU] vec * scal failed\n");
    }

    float test_dot = a.dot(b); 
    if (!(test_dot == 11.0f)) {
        printf("[GPU] dot failed\n");
    }

    float test_norm = b.norm();
    if (!(test_norm == 5.0f)) {
        printf("[GPU] norm failed\n");
    }

    float test_sqnorm = b.sqnorm();
    if (!(test_sqnorm == 25.0f)){
        printf("[GPU] sqnorm failed\n");
    }

    Vec2f test_unit = b.unit();
    if (!(test_unit[0] == 3.0f/5.0f && test_unit[1] == 4.0f/5.0f)){
        printf("[GPU] unit failed\n");
    }
};

#endif


int main(void) {

    Vec2f a({1.0f, 2.0f});
    Vec2f b({3.0f, 4.0f});

    Vec2f test_vec_add = a + b;
    if (!(test_vec_add[0] == 4.0f && test_vec_add[1] == 6.0f)) {
        printf("[CPU] vec + vec failed\n");
    }

    Vec2f test_scal_add_left = 1.0f + a;
    if (!(test_scal_add_left[0] == 2.0f && test_scal_add_left[1] == 3.0f)) {
        printf("[CPU] scalar + vec failed\n");
    }

    Vec2f test_scal_add_right = a + 1.0f;
    if (!(test_scal_add_right[0] == 2.0f && test_scal_add_right[1] == 3.0f)) {
        printf("[CPU] vec + scalar failed\n");
    }

    Vec2f test_vec_sub = a - b;
    if (!(test_vec_sub[0] == -2.0f && test_vec_sub[1] == -2.0f)){
        printf("[CPU] vec - vec failed\n");
    }

    Vec2f test_scal_mult_right = a * 2.0f;
    if (!(test_scal_mult_right[0] == 2.0f && test_scal_mult_right[1] == 4.0f)){
        printf("[CPU] vec * scal failed\n");
    }

    float test_dot = a.dot(b); 
    if (!(test_dot == 11.0f)) {
        printf("[CPU] dot failed\n");
    }

    float test_norm = b.norm();
    if (!(test_norm == 5.0f)) {
        printf("[CPU] norm failed\n");
    }

    float test_sqnorm = b.sqnorm();
    if (!(test_sqnorm == 25.0f)){
        printf("[CPU] sqnorm failed\n");
    }

    Vec2f test_unit = b.unit();
    if (!(test_unit[0] == 3.0f/5.0f && test_unit[1] == 4.0f/5.0f)){
        printf("[CPU] unit failed\n");
    }

#ifdef __CUDACC__
    kernel_test<<<1, 1>>>();
    cudaDeviceSynchronize();
#endif

    return 0;
}
