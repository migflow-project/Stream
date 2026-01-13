#ifndef __STREAM_DIRECTALPHASHAPE3D__
#define __STREAM_DIRECTALPHASHAPE3D__

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
    struct AlphaShape3D;
} // namespace stream::mesh

// C-compatible typedefs and interface
#ifdef __cplusplus 
extern "C" {
#endif

    typedef struct stream::mesh::AlphaShape3D AlphaShape3D;

    // Create and destroy a pointer to an AlphaShape3D structure
    AlphaShape3D* AlphaShape3D_create();
    void AlphaShape3D_destroy(AlphaShape3D* ashape);

    // Given @nnodes 3D coords in row-major order (x0 y0 z0 x1 y1 z1 ...) and @nnodes alpha values
    // Set the corresponding point cloud and desired alphas
    void AlphaShape2D_set_nodes(AlphaShape3D* const ashape, uint32_t nnodes, fp_tt const* const coords, fp_tt const * const alpha);

    // Init the alpha-shape (allocate memory, precompute number of neighbors)
    void AlphaShape3D_init(AlphaShape3D* const ashape);

    // Compute the alpha-shape
    void AlphaShape3D_compute(AlphaShape3D* const ashape);

    // Retrieve the number of element in the alpha-shape
    uint32_t AlphaShape3D_get_nelem(AlphaShape3D const * const ashape);

    // Retrieve the elements in the alpha-shape
    void AlphaShape3D_get_elem(AlphaShape3D const * const ashape, uint32_t * const elems);

    void AlphaShape3D_get_ordered_nodes(AlphaShape3D * const ashape, fp_tt * const nodes);
#ifdef __cplusplus 
}
#endif

namespace stream::mesh {

struct AlphaShape3D {
    // Raw primitive
    using Prim = geo::Tet<fp_tt, 3>;
    // Globally indexed primitive
    using Elem = geo::TetElem<fp_tt, 3, uint32_t>;
    // Locally indexed primitive (assume < 256 neighbors)
    using LocalElem = geo::TriElem<fp_tt, 3, uint8_t>;

    using VecT = typename geo::Vec<fp_tt, 3>;   // 3d vector type
    using BBoxT = typename geo::BBox<fp_tt, 3>; // 3d bbox type

    static uint32_t constexpr dim = 3;      // Dimension of the problem
    static uint32_t constexpr n_inf_pts = 4;  // Number of infinity points (one above, one below right, one below left)
    static uint32_t constexpr n_init_tri = 4; // Number of initial triangles (linking all infty points to the current node)
    static constexpr uint32_t const WARPSIZE = 32; // size of warp
                                              
    // Compute the internal bounding boxes of vertices inside the LBVH when 
    // fed as spheres (center, radius)
    struct InternalNodeDataFunctor {
        BBoxT __host__ __device__ init(const Sphere3D& prim) const noexcept {
            return BBoxT(prim.c, prim.c);   // Bounding box of a point
        }

        void __host__ __device__ combine(BBoxT& lhs, const BBoxT& rhs) const noexcept {
            lhs.combineBox(rhs); // Merge 2 bboxes
        }

        BBoxT __host__ __device__ finalize(const BBoxT& data) const noexcept {
            return data;
        }
    };

    // Karras LBVH taking 3D spheres as objects (center, radius) 
    // This structure is used to accelerate the radius searches.
    using NodeLBVH3D = stream::geo::LBVH<Sphere3D, InternalNodeDataFunctor, 3>;

    uint32_t n_points; // number of nodes in the mesh
    uint32_t n_neig;   // number of neighbors (sum of local neighborhoods) 
    uint32_t n_edges;  // number of edges (= filtered neighbors)
    uint32_t n_tri;    // number of tri
    uint32_t n_blocks; // Number of blocks for the hybrid ELL/CSR
    
    // Temp memory for the CUB calls
    size_t temp_mem_size = 0;
    AvaDeviceArray<char, size_t>::Ptr temp_mem;

    AvaDeviceArray<Sphere3D, int>::Ptr d_coords;       // coordinates of points

    AvaDeviceArray<uint32_t, int>::Ptr d_node_nineig; // Number of initial neighbors per node
    AvaDeviceArray<uint8_t, int>::Ptr  d_active_neig; // Which of the initial neighbors are connected the the node

    // Input and temporary buffers for the computation of the alpha-shape.
    // These arrays are stored in a custom ELL/CSR format to increase GPU coalescent accesses
    AvaDeviceArray<uint32_t, int>::Ptr  d_block_offset; // Offset of the i-th block of the ELL/CSR representation
    AvaDeviceArray<uint32_t, int>::Ptr  d_row_offset;   // Offset of the i-th node data in d_node_triloc/d_node_neig
    AvaDeviceArray<LocalElem, int>::Ptr d_node_triloc;  // Local triangle representation using local neighbors indices 
    AvaDeviceArray<uint32_t, int>::Ptr  d_node_neig;     // Input global neighbor ID
    AvaDeviceArray<uint8_t, int>::Ptr   d_node_ntri_out; // How many triangles should the i-th node output 
                                       // (because we store each triangle once per node, only one node 
                                       // should output the triangle)
    AvaDeviceArray<uint8_t, int>::Ptr d_node_ntri; // How many triangles should the i-th node output 

    AvaDeviceArray<uint32_t, int>::Ptr d_node_nfneig; // Number of final neighbors per node
    // Output buffers for the alpha-shape (contains the result)
    // These arrays are stored in classic CSR format.
    AvaDeviceArray<uint32_t, int>::Ptr d_row;     // Offset of the i-th node data in d_triglob/d_neig/d_edge_is_bnd
    AvaDeviceArray<uint32_t, int>::Ptr d_trirow;  // Offset of the i-th node data in d_triglob/d_neig/d_edge_is_bnd
    AvaDeviceArray<Elem, int>::Ptr     d_triglob; // Triangle representation with global neighbors ID
    AvaDeviceArray<uint32_t, int>::Ptr d_neig;    // Output global neighbor ID (= neighbors filtered by alpha-shape)
    AvaDeviceArray<uint8_t, int>::Ptr  d_node_is_bnd;  // Is the node on the free-surface boundary ?
    AvaDeviceArray<uint8_t, int>::Ptr  d_edge_is_bnd;  // Is the edges part of the free-surface boundary ?
                                      
    NodeLBVH3D lbvh;  // The spatial search structure

    // Empty initializer
    AlphaShape3D();
    ~AlphaShape3D() = default;

    // Set the point cloud of the alpha-shape
    void set_nodes(const AvaHostArray<Sphere3D, int>::Ptr nodes);

    // Compute LBVH and initialize all arrays with exact number of neighbors
    void init();

    // Compute the alpha shape
    void compute();

    // Compress the gpu arrays before outputting to CPU
    void compress(); 

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
    // @tri [in/out] : user provided vector filled with the triangles. Output size is #tri
    uint32_t getTri(std::vector<Elem>& tri) const;

    // Get every edge of the alphashape. The output is a CSR array representing the adjacency matrix of the underlying graph.
    // @returns : the number of edges 
    // @warning : The IDs are a permutation of the original IDs. 
    //            To get the permutation, use getPermutation();
    // @nEdgeNodes [in/out] : user provided vector filled with the starting index of the edges for the i-th node 
    // @edges [in/out] : user provided vector filled with the edges
    uint32_t getEdge(std::vector<uint32_t>& nEdgeNodes, std::vector<uint32_t>& edges) const;

    // Return whether or not edges are on the boundary
    // @returns : the number of edges 
    // @warning : The IDs are a permutation of the original IDs. 
    //            To get the permutation, use getPermutation();
    // @isBoundary [in/out] : user provided vector filled with one boolean per edges. 
    //                        false if edge is not boundary and true otherwise.
    uint32_t getBoundaryEdges(std::vector<uint8_t>& _isBoundaryEdge) const;

    // Return whether or not the nodes are on the boundary 
    // @returns : the number of nodes 
    // @warning : The indices are a permutation of the original indices
    //            To get the permutation, use getPermutation();
    // @d_node_is_bnd [in/out] : user provided vector filled with one boolean per node.
    //                            false if node is not boundary and true otherwise.
    uint32_t getBoundaryNodes(std::vector<uint8_t>& _isBoundaryNode) const;

    // Return the coordinates sorted by their Morton code
    // @returns : the number of nodes 
    // @coords_m [in/out] : user provided vector filled with one boolean per node.
    //                           false if node is not boundary and true otherwise.
    uint32_t getCoordsMorton(std::vector<Sphere3D>& coords_m) const;

    // Temporary local triangulation structure to send in the kernels.
    // It allows to avoid sending *this to kernels.
    struct TriLoc {
        const uint32_t n_points;
        AvaView<uint32_t, -1> d_block_offset_v;
        AvaView<LocalElem, -1> d_node_triloc_v;
        AvaView<uint32_t, -1> d_node_neig_v;

        // Return a new TriLoc with memory offsets corresponding 
        // to the start of the nodes/neighbor of this thread's local 
        // triangulation in the global memory buffers
        __device__ inline TriLoc thread_init(uint32_t const tid) const {
            uint32_t const elem_offset = get_elem_offset(tid);
            uint32_t const neig_offset = get_neig_offset(tid);

            LocalElem * elem = d_node_triloc_v.data + elem_offset;
            uint32_t * neig = d_node_neig_v.data + neig_offset;

            const int elem_shape[1] = {d_node_triloc_v.dyn_shape[0]};
            const int node_shape[1] = {d_node_neig_v.dyn_shape[0]};

            return {
                n_points, 
                d_block_offset_v,
                AvaView<LocalElem, -1>(elem, elem_shape),
                AvaView<uint32_t, -1>(neig, node_shape)
            };
        }

        // Return the memory d_row_offset of tid's buffer in tri
        __device__ inline uint32_t get_elem_offset(uint32_t const tid) const {
            // return 128*(tid/WARPSIZE)*WARPSIZE + tid % WARPSIZE;
            return d_block_offset_v(tid / WARPSIZE) + tid % WARPSIZE + n_init_tri*(tid/WARPSIZE)*WARPSIZE;
        };

        // Return the memory d_row_offset of tid's buffer in d_neig
        __device__ inline uint32_t get_neig_offset(uint32_t const tid) const {
            return d_block_offset_v(tid / WARPSIZE) + tid % WARPSIZE;
        };

        // Get the j-th element in the local triangulation
        __device__ inline LocalElem& get_elem(uint32_t const j) const {
            return d_node_triloc_v(j*WARPSIZE);
        };

        // Get the j-th neighbor in the local triangulation
        __device__ inline uint32_t get_neig(uint32_t const j) const {
            if (j >= n_inf_pts) {
                return d_node_neig_v((j-n_inf_pts)*WARPSIZE);
            }
            return j + n_points;
        };

        // Set the j-th element in the local triangulation
        __device__ inline uint32_t& set_neig(uint32_t const j) const {
            return d_node_neig_v((j-n_inf_pts)*WARPSIZE);
        };
    };

    __host__ TriLoc get_triloc_struct() const {
        return {
            n_points,
            d_block_offset->to_view<-1>(), 
            d_node_triloc->to_view<-1>(),
            d_node_neig->to_view<-1>()
        };
    }
};

} // namespace stream::mesh

#endif // __STREAM_DIRECTALPHASHAPE3D__
