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
#include "defines.h"
#include "geometry/vec.hpp"
#include "geometry/bbox.hpp"
#include "geometry/primitives.hpp"

namespace stream::geo {
    template struct Sphere<fp_tt, 2>;
    template struct Sphere<fp_tt, 3>;
    template struct Edge<fp_tt, 2>;
    template struct Edge<fp_tt, 3>;
    template struct Tri<fp_tt, 2>;
    template struct Tri<fp_tt, 3>;
    template struct Tet<fp_tt, 3>; // Cannot have Tet in 2D
                                 
    template struct SphereElem<fp_tt, 2>;
    template struct SphereElem<fp_tt, 3>;
    template struct EdgeElem<fp_tt, 2>;
    template struct EdgeElem<fp_tt, 3>;
    template struct TriElem<fp_tt, 2>;
    template struct TriElem<fp_tt, 3>;
    template struct TetElem<fp_tt, 3>; // Cannot have Tet in 2D
    
    template struct BBox<fp_tt, 2>;
    template struct BBox<fp_tt, 3>;
} // namespace stream::geo
