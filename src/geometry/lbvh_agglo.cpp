#include "lbvh_agglo.hpp"

template struct stream::geo::LBVHa<Vec2f, stream::geo::ComputeBB_Functor<Vec2f>, 2>;
template struct stream::geo::LBVHa<Sphere2D, stream::geo::ComputeBB_Functor<Sphere2D>, 2>;
template struct stream::geo::LBVHa<Edge2D, stream::geo::ComputeBB_Functor<Edge2D>, 2>;
template struct stream::geo::LBVHa<Tri2D, stream::geo::ComputeBB_Functor<Tri2D>, 2>;

template struct stream::geo::LBVHa<Sphere3D, stream::geo::ComputeBB_Functor<Sphere3D>, 3>;
