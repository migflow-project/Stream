#ifndef __STREAM_MESH_HPP__
#define __STREAM_MESH_HPP__


#include <type_traits>
#include "defines.h"
#include "ava_device_array.h"
#include "lbvh_agglo.hpp"
#include "vec.hpp"
#include "primitives.hpp"
#include "predicates.hpp"

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// forward definitions 
namespace stream::mesh {
    
    template<int dim, uint32_t block_size = WARP_SIZE> struct Mesh;

} // namespace stream::mesh

// C-compatible typedefs and interface
#ifdef __cplusplus
extern "C" {
#endif

    typedef struct stream::mesh::Mesh<2> Mesh2D;
    typedef struct stream::mesh::Mesh<3> Mesh3D;


    // Create an empty 2D mesh
    Mesh2D* Mesh2D_create();

    // Destroy a 2D mesh
    void Mesh2D_destroy(Mesh2D* mesh);

    // Set the nodes of the 2D mesh
    void Mesh2D_set_nodes(Mesh2D* mesh, fp_tt const * const coords);

    // Initialize the mesh
    void Mesh2D_init(Mesh2D* mesh);

    // Compute the mesh
    void Mesh2D_compute(Mesh2D* mesh);

    // Compress the mesh 
    void Mesh2D_compress(Mesh2D* mesh);


#ifdef __cplusplus
}
#endif

namespace stream::mesh {

    template<int dim, uint32_t block_size>
    struct Mesh {
        // Raw primitive
        using Prim = typename std::conditional<dim==2, geo::Tri<fp_tt, 2>, geo::Tet<fp_tt, 3>>::type;   
        // Globally indexed primitive
        using Elem = typename std::conditional<dim==2, geo::TriElem<fp_tt, 2, uint32_t>, geo::TetElem<fp_tt, 3, uint32_t>>::type;   
        // Locally indexed primitive for first computation phase (assume < 256 neighbors)
        using LocalElem = typename std::conditional<dim==2, geo::EdgeElem<fp_tt, 2, uint8_t>, geo::TriElem<fp_tt, 3, uint8_t>>::type;   
        // Locally indexed primitive for recovery phase (assume < MAX_UINT32 neigbors)
        using RecoveryLocalElem = typename std::conditional<dim==2, geo::EdgeElem<fp_tt, 2, uint32_t>, geo::TriElem<fp_tt, 3, uint32_t>>::type;   
        using VecT = typename geo::Vec<fp_tt, dim>;   // 2d/3d vector type
        using BBoxT = typename geo::BBox<fp_tt, dim>; // 2d/3d bbox type

        // CONSTANT : cannot be modified
        // Number of infinity nodes and initial simplices in the triangulation.
        // In the classical Bowyer-Watson algorithm, there are generally dim+1
        // infinity nodes and 1 simplex containing the whole point set. But here 
        // we already insert 1 point in the triangulation, hence splitting the 
        // initial simplex in dim+1 new simplices.
        static constexpr uint32_t const n_inf_nodes = dim+1;
        static constexpr uint32_t const n_init_elem = n_inf_nodes;

        // PARAMETER : can be modified (must be < 256)
        // Guess the number of neighbors of each node in the mesh. 
        // This is used to allocate the memory of the edge array. 
        // A good value for this parameter ensures that the majority of the 
        // nodes are processed in the first triangulation phase without too much cost
        static constexpr uint32_t const n_neig_guess = (dim == 2) ? 32 : 32;

        // PARAMETER : can be modifier
        // Number of morton neighbors to insert in the initial triangulation.
        // This is done as an attempt to reduce the size of the circumspheres 
        // of the simplices in the initial triangulation.
        // A good value for this parameter ensures that only a few insertions
        // are needed afterwards.
        static constexpr uint32_t const n_init_insert = (dim == 2) ? 10 : 10;

        // CONSTANT : cannot be modified
        // Maximum number of local simplices given initial guess of number of 
        // neighbors. In 2D the maximum number of tri is given by a triangle 
        // loop around the node (n tri). In 3D the maximum number of tet is 
        // given by a ball around the node, whose surface is a planar triangulation 
        // with a maximum number of triangles equal to 2 times its number of nodes.
        static constexpr uint32_t const n_init_local_elem = (dim == 2) ? n_neig_guess : 2*n_neig_guess;

        uint32_t n_nodes; // Number of nodes in the mesh 
        uint32_t n_edges; // Number of edges in the mesh 
        uint32_t n_elems; // Number of simplices in the mesh
        uint32_t n_blocks; // Number of blocks for the hybrid ELL/CSR 
        uint32_t cur_max_nneig; // Current maximum number of neighbors
        uint32_t cur_max_nelem; // Current maximum number of simplices in the local triangulation
        
        // Temporary memory for CUB calls
        size_t temp_mem_size;
        AvaDeviceArray<uint8_t, size_t>::Ptr d_temp_mem;

        // Pointer to the positions of the nodes. 
        // This can be updated in-between triangulations without 
        // the need to instanciate another Mesh object
        typename AvaDeviceArray<VecT, int>::Ptr d_nodes;

        // Local triangulations around each node for first computation phase
        typename AvaDeviceArray<LocalElem, int>::Ptr d_elemloc;

        // Flag for completed node
        // - 1 if all simplices in the local triangulation is locally Delaunay 
        // - 0 otherwise
        AvaDeviceArray<uint8_t, int>::Ptr d_node_is_complete;

        // Number of local simplices in the local triangulation
        AvaDeviceArray<uint32_t, int>::Ptr d_node_nelemloc;

        // Number of local neighbors in the local triangulation
        AvaDeviceArray<uint32_t, int>::Ptr d_node_nneigloc;

        // Number of simplices output by each node (to avoid triangles duplication 
        // in d_triglob)
        AvaDeviceArray<uint32_t, int>::Ptr d_node_nelem_out;

        // Global triangulation
        typename AvaDeviceArray<Elem, int>::Ptr d_elemglob;

        // Neighbor list of each node
        AvaDeviceArray<uint32_t, int>::Ptr d_neig;

        // Spatial search structure 
        geo::LBVHa<VecT, geo::ComputeBB_Functor<VecT>, dim> lbvh;


        // Empty initializer
        Mesh() noexcept;
        ~Mesh() = default;

        // - Compute the spatial search structure 
        // - initialize all arrays based on the size of d_nodes
        void init(void);

        void insert_morton_neighbors(void);
        
        void insert_quadrant_neighbors(void);
        void insert_BVH_neighbors(void);
        void insert_by_circumsphere_checking(void);

        void remove_super_nodes(void);

        // Compress the local triangulations into a global one, without 
        // simplex duplication. 
        // WARNING: The indices of the tuples are based on the morton-ordering
        //          and NOT on the initial point order. This is done for 
        //          efficiency purposes.
        // This should be used e.g. for outputting triangulation to CPU or 
        // for visualisation purposes.
        void compress_into_global(void);

        // Rearrange an array from the initial ordering towards the morton order
        template<typename T>
        void permuteForward(typename AvaDeviceArray<T, int>::Ptr d_in, typename AvaDeviceArray<T, int>::Ptr d_out) const;

        // Rearrange an array from the morton ordering towards the initial order
        template<typename T>
        void permuteBackward(typename AvaDeviceArray<T, int>::Ptr d_in, typename AvaDeviceArray<T, int>::Ptr d_out) const;

        // Temporary local triangulation structure to send in the kernels.
        // It allows to avoid sending *this to kernels.
        struct TriLoc {
            const uint32_t n_nodes;
            const uint32_t cur_max_nneig;
            const uint32_t cur_max_nelem;
            AvaView<LocalElem, -1> d_node_elemloc_v;
            AvaView<uint32_t, -1> d_neig_v;

            // Get the starting index of the triangle block
            __device__ inline uint32_t get_elem_offset(uint32_t const tid) const {
                return cur_max_nelem*block_size*(tid/block_size) + tid % block_size;
                //     |--------Block size----||--block id----|  |-Column in block-|
            };

            // Get the starting index of the neighbor block
            __device__ inline uint32_t get_neig_offset(uint32_t const tid) const {
                return cur_max_nneig*((tid/block_size)*block_size) + tid % block_size;
            };

            // Get the j-th triangle
            __device__ inline LocalElem& get_elem(uint32_t const elem_offset, uint32_t const j) const {
                return d_node_elemloc_v(elem_offset + j*block_size);
                //                                  |row in block|
            };

            // Get the j-th local neighbor
            __device__ inline uint32_t get_neig(uint32_t const neig_offset, uint32_t const j) const {
                // Infinity points are indexed as 
                // - 0, 1, 2, (3 in dim3) locally
                // - n_nodes + i, i=0, 1, 2, (3 in dim3) globally
                if (j >= n_inf_nodes) {
                    return d_neig_v(neig_offset + (j-n_inf_nodes)*block_size);
                }
                return j + n_nodes; // Return global index of infinity point
            };

            // Set the j-th local neighbor
            __device__ inline uint32_t& set_neig(uint32_t const neig_offset, uint32_t const j) const {
                // Error if called on an infinity node
                return d_neig_v(neig_offset + (j-n_inf_nodes)*block_size);
            };
        };

        __host__ TriLoc get_triloc_struct() const {
            return {
                n_nodes,
                cur_max_nneig,
                cur_max_nelem,
                d_elemloc->template to_view<-1>(),
                d_neig->to_view<-1>()
            };
        }

    };

    template<typename T>
    __device__ inline fp_tt incircle_SoS(
            uint32_t i1,    // tri.a   (= node of current thread)
            uint32_t i2,    // tri.b
            uint32_t i3,    // tri.c
            uint32_t i4,    // inserted node
            const AvaView<T, -1> nodes  // global node array
            ) {
        T pa = nodes(i1);
        T pb = nodes(i2);
        T pc = nodes(i3);
        T pd = nodes(i4);

        fp_tt const orientation = orient2d(&pa[0], &pb[0], &pc[0]);

        fp_tt errbound;
        fp_tt det = incircle(&pa[0], &pb[0], &pc[0], &pd[0], &errbound);

        if (!((det > errbound) || (-det > errbound))) {
            bool swap = false;
            uint32_t tmp;
            T vtmp;
            if (i1 > i3) { tmp = i3; i3 = i1; i1 = tmp; vtmp = pc; pc = pa; pa = vtmp; swap = !swap; }
            if (i2 > i4) { tmp = i4; i4 = i2; i2 = tmp; vtmp = pd; pd = pb; pb = vtmp; swap = !swap; }
            if (i1 > i2) { tmp = i2; i2 = i1; i1 = tmp; vtmp = pb; pb = pa; pa = vtmp; swap = !swap; }
            if (i3 > i4) { tmp = i4; i4 = i3; i3 = tmp; vtmp = pd; pd = pc; pc = vtmp; swap = !swap; }
            if (i2 > i3) { tmp = i3; i3 = i2; i2 = tmp; vtmp = pc; pc = pb; pb = vtmp; swap = !swap; }

            det = incircleexact(&pa[0], &pb[0], &pc[0], &pd[0]);
            int depth = 0;
            while (det == 0.0f){
                depth++;
                switch (depth){
                    case 1: det = + orient2d(&pb[0], &pc[0], &pd[0]); break;
                    case 2: det = - orient2d(&pa[0], &pc[0], &pd[0]); break;
                    default:
                            // The 4 points are colinear, do not connect them
                            det = (swap + (orientation < 0.0f)) == 1 ? 1.0f : -1.0f;
                            break;
                }
            }

            if (swap) det = -det;
        }

        if (orientation < 0.0f) det = -det;
        return det;
    }
} // namespace stream::mesh


#endif // __STREAM_MESH_HPP__
