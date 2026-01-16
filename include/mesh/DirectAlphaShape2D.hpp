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
#ifndef __STREAM_DIRECTALPHASHAPE2D__
#define __STREAM_DIRECTALPHASHAPE2D__

#include <stdexcept>
#include <vector>

#include "defines.h"
#include "ava_device_array.h"
#include "ava_host_array.h"
#include "primitives.hpp"
#include "predicates.hpp"
#include "lbvh.hpp"
#include "defines.h"

// Forward declarations
namespace stream::mesh {
    struct AlphaShape2D;
} // namespace stream::mesh


// C-compatible typedefs and interface
#ifdef __cplusplus 
extern "C" {
#endif

    typedef struct stream::mesh::AlphaShape2D AlphaShape2D;

    // Create and destroy a pointer to an AlphaShape2D structure
    AlphaShape2D* AlphaShape2D_create();
    void AlphaShape2D_destroy(AlphaShape2D* ashape);

    // Given @nnodes 2D coords in row-major order (x0 y0 x1 y1 ...) and @nnodes alpha values
    // Set the corresponding point cloud and desired alphas
    void AlphaShape2D_set_nodes(AlphaShape2D* const ashape, uint32_t nnodes, fp_tt const* const coords, fp_tt const * const alpha);

    // Init the alpha-shape (allocate memory, precompute number of neighbors)
    void AlphaShape2D_init(AlphaShape2D* const ashape);

    // Compute the alpha-shape
    void AlphaShape2D_compute(AlphaShape2D* const ashape);

    // Retrieve the number of element in the alpha-shape
    uint32_t AlphaShape2D_get_nelem(AlphaShape2D const * const ashape);

    // Retrieve the elements in the alpha-shape
    void AlphaShape2D_get_elem(AlphaShape2D const * const ashape, uint32_t * const elems);

    void AlphaShape2D_get_ordered_nodes(AlphaShape2D * const ashape, fp_tt * const nodes);
#ifdef __cplusplus 
}
#endif

namespace stream::mesh {

struct AlphaShape2D {
    // Raw primitive
    using Prim = geo::Tri<fp_tt, 2>;
    // Globally indexed primitive
    using Elem = geo::TriElem<fp_tt, 2, uint32_t>;
    // Locally indexed primitive (assume < 256 neighbors)
    using LocalElem = geo::EdgeElem<fp_tt, 2, uint8_t>;

    using VecT = typename geo::Vec<fp_tt, 2>;   // 2d vector type
    using BBoxT = typename geo::BBox<fp_tt, 2>; // 2d bbox type

    // Compute the internal bounding boxes of vertices inside the LBVH when 
    // fed as spheres (center, radius)
    struct InternalNodeDataFunctor {
        BBoxT __host__ __device__ init(const Sphere2D& prim) const noexcept {
            return BBoxT(prim.c, prim.c);   // Bounding box of a point
        }

        void __host__ __device__ combine(BBoxT& lhs, const BBoxT& rhs) const noexcept {
            lhs.combineBox(rhs); // Merge 2 bboxes
        }

        BBoxT __host__ __device__ finalize(const BBoxT& data) const noexcept {
            return data;
        }
    };

    // Karras LBVH taking 2D spheres as objects (center, radius) 
    // This structure is used to accelerate the radius searches.
    using NodeLBVH2D = stream::geo::LBVH<Sphere2D, InternalNodeDataFunctor, 2>;
                                                      
    static constexpr uint32_t const dim = 2;        // Dimension of the problem
    static constexpr uint32_t const n_inf_nodes = 3;  // Number of infinity points (one above, one below right, one below left)
    static constexpr uint32_t const n_init_elem = 3; // Number of initial triangles (linking all infinity points to the current node)

    static constexpr uint32_t const WARPSIZE = 32; // size of warp

    uint32_t n_nodes;    // Number of nodes in the mesh
    uint32_t n_neig;     // Number of neighbors (sum of local neighborhoods) 
    uint32_t n_edges;    // Number of edges (= filtered neighbors)
    uint32_t n_elems;    // Number of tri
    uint32_t n_blocks;   // Number of blocks for the hybrid ELL/CSR
                     
    // Temp memory for the CUB calls
    size_t temp_mem_size = 0;
    AvaDeviceArray<char, size_t>::Ptr d_temp_mem;

    // Coordinates of nodes and their alpha values stored as radius
    AvaDeviceArray<Sphere2D, int>::Ptr d_coords;
    // Number of potential neighbor for each node. size = n_points
    AvaDeviceArray<uint32_t, int>::Ptr d_node_nineig; 
    // Indicate which potential neighbors are connected in the triangulation 
    AvaDeviceArray<uint8_t, int>::Ptr d_active_neig; 

    // Number of elements output by a node. Because we store each triangle once
    // per node, only one node should output it.
    AvaDeviceArray<uint8_t, int>::Ptr d_node_nelem_out; 
    // Number of local triangles of each node
    AvaDeviceArray<uint8_t, int>::Ptr d_node_nelem;
    // Number of final neighbors per node
    AvaDeviceArray<uint32_t, int>::Ptr d_node_nfneig;

    /*
     * The following arrays are stored in a custom ELL/CSR format to increase 
     * GPU coalescent accesses. 
     *
     * It increases the total memory requirement of irregular triangulations but
     * decreases runtime.
     * They should be used for computations.
     * 
     * The hybrid ELL/CSR format groups blocks of 32 rows in the ELL format 
     * and expresses the blocks in CSR format. 
     * The @d_block_offset array gives the memory offset of each block
     *     @d_row_offset array gives the memory offset of each row
     *
     * E.g considering blocks of 2 row, the matrix :
     *         0  1  0  0  1  0  0
     *         1  1  0  0  0  0  0 
     *         1  0  0  1  0  0  1
     *         0  0  1  0  1  0  0
     *         0  0  0  0  0  1  0
     *         1  1  0  1  0  0  0
     * Can be expressed in 3 blocks 
     *             values                 columns
     *  block 0    1  1                   1  4
     *             1  1                   0  1
     *
     *  block 1    1  1  1                0  3  6
     *             1  1  x                2  4  x
     *
     *  block 2    1  x  x                5  x  x
     *             1  1  1                0  1  3
     *
     *  d_block_offset = [ 0, 4, 10, 16]
     *  d_row_offset = [0, 2, 4, 7, 10, 13, 16]
     */

    // Offset of i-th block of ELL/CSR 
    AvaDeviceArray<uint32_t, int>::Ptr  d_block_offset;
    // Offset of i-th node data 
    AvaDeviceArray<uint32_t, int>::Ptr  d_row_offset;
    // Local triangulation representation 
    AvaDeviceArray<LocalElem, int>::Ptr d_node_elem;    
    // Global neighbors ID of local neighborhoods.
    AvaDeviceArray<uint32_t, int>::Ptr  d_node_neig;      
                                    
    /*
     * Output buffers for the alpha-shape. These arrays are stored in classic
     * CSR format.
     * They should be used for Host-Device communications.
     */

    // Offset of the i-th node neighbor data
    AvaDeviceArray<uint32_t, int>::Ptr d_row;
    // Global neighbors ID. 
    // See @d_row to get index of local neighborhood of i-th node
    AvaDeviceArray<uint32_t, int>::Ptr d_neig;  
    // Offset of the i-th node triangle data 
    AvaDeviceArray<uint32_t, int>::Ptr d_elemrow;
    // Global triangulation. See @d_trirow to get the index of the 
    // local triangulation of i-th node.
    AvaDeviceArray<Elem, int>::Ptr     d_elemglob;

    // Is the i-th node on the free-surface boundary ?
    AvaDeviceArray<uint8_t, int>::Ptr  d_node_is_bnd;  
    // Is the i-th edge part of the free-surface boundary ?
    AvaDeviceArray<uint8_t, int>::Ptr  d_edge_is_bnd;  

    NodeLBVH2D lbvh;   // Spatial search structure
                  
    // Empty initializer
    AlphaShape2D();
    ~AlphaShape2D() = default;

    // Set the point cloud of the alpha-shape
    void set_nodes(const AvaHostArray<Sphere2D, int>::Ptr nodes);

    // Compute LBVH and initialize all arrays with exact number of neighbors
    void init();

    // Compute the alpha shape
    void compute();

    // Compress the gpu arrays before outputting to CPU
    void compress(); 

    // From initial ordering to morton ordering
    template<typename T>
    void permuteForward(AvaDeviceArray<T, int>::Ptr d_in, AvaDeviceArray<T, int>::Ptr d_out) const {
        AvaView<T, -1> d_in_v = d_in->template to_view<-1>();
        AvaView<T, -1> d_out_v = d_out->template to_view<-1>();
        AvaView<uint32_t, -1> d_map_v = lbvh.d_map_sorted->to_view<-1>();

        ava_for<256>(nullptr, 0, n_nodes, [=] __device__ (uint32_t const tid) {
            d_out_v(tid) = d_in_v(d_map_v(tid));
        });
    }

    // From morton ordering to initial ordering
    template<typename T>
    void permuteBackward(AvaDeviceArray<T, int>::Ptr d_in, AvaDeviceArray<T, int>::Ptr d_out) const {
        AvaView<T, -1> d_in_v = d_in->template to_view<-1>();
        AvaView<T, -1> d_out_v = d_out->template to_view<-1>();
        AvaView<uint32_t, -1> d_map_v = lbvh.d_map_sorted->to_view<-1>();

        ava_for<256>(nullptr, 0, n_nodes, [=] __device__ (uint32_t const tid) {
            d_out_v(d_map_v(tid)) = d_in_v(tid);
        });
    }

    // Get the permutation of the coordinates indices.
    // @returns : the number of indices (= number of points)
    // @perm [in/out] : user provided vector filled with the permutation. 
    //                  If x_i is the initial ordering and x_r is the reordered 
    //                  vector, we have : x_i = x_r[perm[i]]
    uint32_t getPermutation(std::vector<uint32_t>& perm) const;

    // Get the triangulation of the alphashape. 
    // @warning : The IDs are a permutation of the original IDs. 
    //            To get the permutation, use getPermutation();
    // @returns : the number of triangles.
    // @tri [in/out] : user provided vector filled with the triangles. Output size is 3 x #tri
    uint32_t getElem(std::vector<Elem>& tri) const;

    // Get every edge of the alphashape. The output is a CSR array representing the adjacency matrix of the underlying graph.
    // @returns : the number of edges 
    // @warning : The IDs are a permutation of the original IDs. 
    //            To get the permutation, use getPermutation();
    // @nEdgeNodes [in/out] : user provided vector filled with the starting index of the edges for the i-th node 
    // @edges [in/out] : user provided vector filled with the edges
    uint32_t getEdge(std::vector<uint32_t>& nEdgeNodes, std::vector<uint32_t>& edges) const;

    // Return whether or not the nodes are on the boundary 
    // @returns : the number of nodes 
    // @warning : The indices are a permutation of the original indices
    //            To get the permutation, use getPermutation();
    // @node_is_bnd [in/out] : user provided vector filled with one boolean per node.
    //                            false if node is not boundary and true otherwise.
    uint32_t getBoundaryNodes(std::vector<uint8_t>& node_is_bnd) const;

    // Return the coordinates sorted by their Morton code
    // @returns : the number of nodes 
    // @coords_m [in/out] : user provided vector filled with one boolean per node.
    //                           false if node is not boundary and true otherwise.
    uint32_t getCoordsMorton(std::vector<Sphere2D>& coords_m) const;

    // Local triangulation T_i around each node. 
    // Each local triangulation is a view in the global memory buffers. 
    struct TriLoc {
        const uint32_t n_points;
        const AvaView<uint32_t, -1> d_block_offset_v;
        AvaView<LocalElem, -1> d_node_elemloc_v;
        AvaView<uint32_t, -1> d_node_neig_v;

        // Return a new TriLoc with memory offsets corresponding 
        // to the start of the nodes/neighbor of this thread's local 
        // triangulation in the global memory buffers
        __device__ inline TriLoc thread_init(uint32_t const tid) const {
            uint32_t const elem_offset = get_elem_offset(tid);
            uint32_t const neig_offset = get_neig_offset(tid);

            LocalElem * elem = d_node_elemloc_v.data + elem_offset;
            uint32_t * neig = d_node_neig_v.data + neig_offset;

            const int elem_shape[1] = {d_node_elemloc_v.dyn_shape[0]};
            const int node_shape[1] = {d_node_neig_v.dyn_shape[0]};

            return {
                n_points, 
                d_block_offset_v,
                AvaView<LocalElem, -1>(elem, elem_shape),
                AvaView<uint32_t, -1>(neig, node_shape)
            };
        }

        // Return the memory offset of tid's buffer in tri
        __device__ inline uint32_t get_elem_offset(uint32_t const tid) const {
            return d_block_offset_v(tid / WARPSIZE) + tid % WARPSIZE + n_init_elem*(tid/WARPSIZE)*WARPSIZE;
        };

        // Return the memory offset of tid's buffer in d_neig
        __device__ inline uint32_t get_neig_offset(uint32_t const tid) const {
            return d_block_offset_v(tid / WARPSIZE) + tid % WARPSIZE;
        };

        // Get the j-th element in the local triangulation
        __device__ inline LocalElem& get_elem(uint32_t const j) const {
            return d_node_elemloc_v(j*WARPSIZE);
        };

        // Get the j-th neighbor in the neighborhood
        __device__ inline uint32_t get_neig(uint32_t const j) const {
            if (j >= n_inf_nodes) {
                return d_node_neig_v((j-n_inf_nodes)*WARPSIZE);
            }
            return j + n_points;
        };

        // Set the j-th neighbor in the neighborhood
        __device__ inline uint32_t& set_neig(uint32_t const j) const {
            return d_node_neig_v((j-n_inf_nodes)*WARPSIZE);
        };
    };

    __host__ TriLoc get_triloc_struct() const {
        return {
            n_nodes,
            d_block_offset->to_view<-1>(), 
            d_node_elem->to_view<-1>(),
            d_node_neig->to_view<-1>()
        };
    }
};

} // namespace stream::mesh

#endif // __STREAM_DIRECTALPHASHAPE2D__
