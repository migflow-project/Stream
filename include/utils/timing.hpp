#ifndef __STREAM_TIMING_H__
#define __STREAM_TIMING_H__

#include <chrono>

// Forward declarations 
namespace stream::utils {
    struct Chrono;
} // namespace stream::utils

namespace stream::utils {
    struct Chrono {
        private:
            std::chrono::time_point<std::chrono::high_resolution_clock> t0;
            std::chrono::time_point<std::chrono::high_resolution_clock> t1;

        public:
            void start(void);
            void stop(void);
            double get_ms(void) const;
    };
} // namespace stream::utils


#endif // __STREAM_TIMING_H__
