#ifndef __STREAM_MORTON_HPP__
#define __STREAM_MORTON_HPP__

#include "defines.h"
#include <cstdint>

namespace stream::geo {
    __host__ __device__ inline uint64_t splitBy3(uint32_t const a) {
        uint64_t x = a & 0x1fffff;  // we only look at the first 21 bits

        // shift left 32 bits, OR with self, and 00011111000000000000000000000000000000001111111111111111
        x  = (x | x << 32) & 0x1f00000000ffff;  
        // shift left 32 bits, OR with self, and 00011111000000000000000011111111000000000000000011111111
        x = (x | x << 16)  & 0x1f0000ff0000ff;  
        // shift left 32 bits, OR with self, and 0001000000001111000000001111000000001111000000001111000000000000
        x = (x | x << 8)   & 0x100f00f00f00f00f;  
        // shift left 32 bits, OR with self, and 0001000011000011000011000011000011000011000011000011000100000000
        x = (x | x << 4)   & 0x10c30c30c30c30c3;  
        x = (x | x << 2)   & 0x1249249249249249;
        return x;
    };

    // Merge bits representation of 3 integers into one.
    __host__ __device__ inline uint64_t encode_magicbits3D(uint32_t const x, uint32_t const y, uint32_t const z) {
        return splitBy3(x) | (splitBy3(y) << 1) | (splitBy3(z) << 2);
    };

    __host__ __device__ inline uint32_t splitBy3_32(uint32_t const a) {
        uint32_t x = a & 0x000003ff ;  // we only look at the first 10 bits
        x = (x | x << 16) & 0x30000ff ;  
        x = (x | x << 8)  & 0x0300f00f;  
        x = (x | x << 4)  & 0x30c30c3 ;  
        x = (x | x << 2)  & 0x9249249 ;
        return x;
    };

    // Merge bits representation of 3 integers into one.
    __host__ __device__ inline uint32_t encode_magicbits3D_32(uint32_t const x, uint32_t const y, uint32_t const z) {
        return splitBy3_32(x) | (splitBy3_32(y) << 1) | (splitBy3_32(z) << 2);
    };

    // Seperate bits from a given integer 2 positions apart
    // eg : 111 becomes 10101
    __host__ __device__ inline uint64_t splitBy2(uint32_t const a) {
        uint64_t x = static_cast<uint64_t>(a);
        x = (x | x << 32) & 0x00000000FFFFFFFF;  
        x = (x | x << 16) & 0x0000FFFF0000FFFF;  
        x = (x | x << 8)  & 0x00FF00FF00FF00FF;  
        x = (x | x << 4)  & 0x0F0F0F0F0F0F0F0F;  
        x = (x | x << 2)  & 0x3333333333333333;
        x = (x | x << 1)  & 0x5555555555555555;
        return x;
    };

    // Merge bits representation of 2 32 bits integers into one 64 bit integer.
    __host__ __device__ inline uint64_t encode_magicbits2D(uint32_t const x, uint32_t const y) {
        return splitBy2(x) | (splitBy2(y) << 1);
    };

    __host__ __device__ inline uint32_t splitBy2_32(uint32_t const a) {
        uint32_t x = a;
        x = (x | x << 16) & 0x0000FFFF;  
        x = (x | x << 8)  & 0x00FF00FF;  
        x = (x | x << 4)  & 0x0F0F0F0F;  
        x = (x | x << 2)  & 0x33333333;
        x = (x | x << 1)  & 0x55555555;
        return x;
    };

    // Merge bits representation of 2 32 bits integers into one 32 bit integer.
    __host__ __device__ inline uint32_t encode_magicbits2D_32(uint32_t const x, uint32_t const y) {
        return splitBy2_32(x) | (splitBy2_32(y) << 1);
    };
} // namespace stream::geo
#endif // __STREAM_MORTON_HPP__
