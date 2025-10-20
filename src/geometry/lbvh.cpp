#include "lbvh.hpp"

template struct stream::geo::LBVH<Vec2f, stream::geo::ComputeBB_Functor<Vec2f>, 2>;
template struct stream::geo::LBVH<Sphere2D, stream::geo::ComputeBB_Functor<Sphere2D>, 2>;
template struct stream::geo::LBVH<Edge2D, stream::geo::ComputeBB_Functor<Edge2D>, 2>;
template struct stream::geo::LBVH<Tri2D, stream::geo::ComputeBB_Functor<Tri2D>, 2>;
