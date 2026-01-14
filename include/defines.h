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
