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
