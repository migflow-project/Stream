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
