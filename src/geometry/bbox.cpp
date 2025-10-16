#include "defines.h"
#include "geometry/bbox.hpp"

namespace stream::geo {

template struct BBox<float, 2>;
template struct BBox<float, 3>;
template struct BBox<float, 4>;

template struct BBox<double, 2>;
template struct BBox<double, 3>;
template struct BBox<double, 4>;

} // namespace stream::geo
