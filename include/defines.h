#ifndef __STREAM_DEFINES_H__
#define __STREAM_DEFINES_H__

// Floating point type
typedef float fp_tt;

// Such that CPU compilation does not warn
#ifndef __device__
#define __device__
#endif

#ifndef __host__
#define __host__
#endif

#ifndef __constant__
#define __constant__
#endif

#ifndef __shared__
#define __shared__
#endif

#ifndef __forceinline__
#define __forceinline__
#endif

#endif // __STREAM_DEFINES_H__
