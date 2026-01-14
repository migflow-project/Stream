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
#include <chrono>
#include "timing.hpp"

namespace stream::utils {
    void Chrono::start() {
        t0 = std::chrono::high_resolution_clock::now(); 
    }
    void Chrono::stop() {
        t1 = std::chrono::high_resolution_clock::now(); 
    }
    double Chrono::get_ms() const {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() * 1e-6;
    }
}
