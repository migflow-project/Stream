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
#include "lbvh.hpp"

template struct stream::geo::LBVH<Vec2f, stream::geo::ComputeBB_Functor<Vec2f>, 2>;
template struct stream::geo::LBVH<Sphere2D, stream::geo::ComputeBB_Functor<Sphere2D>, 2>;
template struct stream::geo::LBVH<Edge2D, stream::geo::ComputeBB_Functor<Edge2D>, 2>;
template struct stream::geo::LBVH<Tri2D, stream::geo::ComputeBB_Functor<Tri2D>, 2>;

template struct stream::geo::LBVH<Sphere3D, stream::geo::ComputeBB_Functor<Sphere3D>, 3>;
