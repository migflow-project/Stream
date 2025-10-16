#include "defines.h"
#include "geometry/vec.hpp"
#include "geometry/bbox.hpp"

namespace stream::geo {

template struct Vec<float, 2>;
template struct Vec<float, 3>;
template struct Vec<float, 4>;

template struct Vec<double, 2>;
template struct Vec<double, 3>;
template struct Vec<double, 4>;

} // namespace stream::geo
