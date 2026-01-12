#include <cstdio>

#include "ava_host_array.h"
#include "defines.h"
#include "DirectAlphaShape3D.hpp"
#include "ava.h"
#include "ava_host_array.hpp"
#include "ava_scan.h"
#include "ava_view.h"
#include "primitives.hpp"
#include "predicates.hpp"
#include "lbvh.hpp"

namespace stream::mesh {

AlphaShape3D::AlphaShape3D() {
    temp_mem = AvaDeviceArray<char, size_t>::create({0});
    d_node_nineig = AvaDeviceArray<uint32_t, int>::create({0});
    d_node_nfneig = AvaDeviceArray<uint32_t, int>::create({0});
    d_node_ntri = AvaDeviceArray<uint8_t, int>::create({0});
    d_node_ntri_out = AvaDeviceArray<uint8_t, int>::create({0});
    d_node_triloc = AvaDeviceArray<LocalElem, int>::create({0});
    d_node_neig = AvaDeviceArray<uint32_t, int>::create({0});
    d_active_neig = AvaDeviceArray<uint8_t, int>::create({0});
    d_node_is_bnd = AvaDeviceArray<uint8_t, int>::create({0});
    d_edge_is_bnd = AvaDeviceArray<uint8_t, int>::create({0});
    d_neig = AvaDeviceArray<uint32_t, int>::create({0});
    d_triglob = AvaDeviceArray<Elem, int>::create({0});
    d_trirow = AvaDeviceArray<uint32_t, int>::create({0});
    d_row = AvaDeviceArray<uint32_t, int>::create({0});
    d_block_offset = AvaDeviceArray<uint32_t, int>::create({0});
    d_row_offset = AvaDeviceArray<uint32_t, int>::create({0});
}

void AlphaShape3D::set_nodes(const AvaHostArray<Sphere3D, int>::Ptr h_nodes) {
    n_points = h_nodes->size();
    d_coords = AvaDeviceArray<Sphere3D, int>::create({(int) n_points});
    d_coords->set(h_nodes);
}

uint32_t AlphaShape3D::getPermutation(std::vector<uint32_t>& perm) const {
    perm.resize(n_points);
    gpu_memcpy(perm.data(), lbvh.d_map_sorted->data, sizeof(perm[0])*n_points, gpu_memcpy_device_to_host);

    return perm.size();
}

uint32_t AlphaShape3D::getTri(std::vector<Elem>& tri) const {
    tri.resize(d_triglob->size);
    gpu_memcpy(tri.data(), d_triglob->data, sizeof(tri[0])*d_triglob->size, gpu_memcpy_device_to_host);
    return tri.size();
}

uint32_t AlphaShape3D::getEdge(std::vector<uint32_t>& nEdgeNodes, std::vector<uint32_t>& edges) const {
    nEdgeNodes.resize(n_points+1);
    edges.resize(n_edges);

    gpu_memcpy(nEdgeNodes.data(), d_row->data, sizeof(nEdgeNodes[0])*(n_points+1), gpu_memcpy_device_to_host);
    gpu_memcpy(edges.data(), d_neig->data, sizeof(edges[0])*(n_edges), gpu_memcpy_device_to_host);

    return edges.size();
}

uint32_t AlphaShape3D::getBoundaryEdges(std::vector<uint8_t>& _isBoundaryEdge) const {
    _isBoundaryEdge.resize(n_edges);
    gpu_memcpy(_isBoundaryEdge.data(), d_edge_is_bnd->data, sizeof(_isBoundaryEdge[0])*(n_edges), gpu_memcpy_device_to_host);
    return _isBoundaryEdge.size();
}

uint32_t AlphaShape3D::getCoordsMorton(std::vector<Sphere3D>& coords_m) const {
    coords_m.resize(n_points);
    gpu_memcpy(coords_m.data(), lbvh.d_obj_m->data, sizeof(coords_m[0])*(n_points), gpu_memcpy_device_to_host);
    return coords_m.size();
}

uint32_t AlphaShape3D::getBoundaryNodes(std::vector<uint8_t>& _isBoundaryNode) const {
    _isBoundaryNode.resize(n_points);
    gpu_memcpy(_isBoundaryNode.data(), d_node_is_bnd->data, sizeof(_isBoundaryNode[0])*(n_points), gpu_memcpy_device_to_host);
    return _isBoundaryNode.size();
}

void AlphaShape3D::init() {

    n_blocks = (n_points + WARPSIZE - 1)/WARPSIZE;
    d_block_offset->resize({(int) n_blocks+1});
    d_row_offset->resize({(int) n_points+1});
    d_row->resize({(int) n_points+1});
    d_node_is_bnd->resize({(int) n_points});
    d_node_ntri_out->resize({(int) n_points});
    d_node_ntri->resize({(int) n_points});
    d_trirow->resize({(int) n_points+1});
    d_node_nineig->resize({(int) n_points});
    d_node_nfneig->resize({(int) n_points});

    // Compute acceleration structure (LBVH)
    lbvh.set_objects(d_coords);
    lbvh.build();

    // Set infinity points at the end of the morton-reordered nodes
    int n_points_v = n_points;
    lbvh.d_obj_m->resize({(int) (n_points+n_inf_pts)});
    const AvaView<Sphere3D, -1> d_coords_m_v = lbvh.d_obj_m->to_view<-1>(); // take view AFTER resize !

    const AvaView<uint32_t, -1> d_node_nineig_v = d_node_nineig->to_view<-1>();
    const AvaView<uint32_t, -1> d_row_offset_v = d_row_offset->to_view<-1>();
    const AvaView<int,     -1> d_internal_sep_v  = lbvh.d_internal_sep->to_view<-1>(); 
    const AvaView<uint8_t, -1> d_child_is_leaf_v = lbvh.d_child_is_leaf->to_view<-1>();
    const AvaView<BBox3D,  -1> d_bboxes_v        = lbvh.d_internal_data->to_view<-1>();
    BBox3D const * const __restrict__ d_bb_glob = lbvh.d_bb_glob->data;

    ava_for<1>(nullptr, 0, 1, [=] __device__(uint32_t const __attribute__((unused)) tid){
        Vec3f const avg = 0.5f*(d_bb_glob->pmax + d_bb_glob->pmin);
        Vec3f const range = d_bb_glob->pmax - d_bb_glob->pmin;
        fp_tt const max_range   = fmaxf(range[0], fmaxf(range[1], range[2]));

        // First vertex of base
        d_coords_m_v(n_points_v) = { 
            .c= {avg[0] - 2*max_range, avg[1] - 2*max_range, avg[2] - 2*max_range},
            .r = 1.0f
        };
        // Second vertex of base
        d_coords_m_v(n_points_v + 1) = {
            .c = {avg[0] + 2*max_range, avg[1] - 2*max_range, avg[2] - 2*max_range},
            .r = 1.0f
        };

        // Third vertex of base
        d_coords_m_v(n_points_v + 2) = {
            .c={avg[0] + 2*max_range, avg[1] + 2*max_range, avg[2] - 2*max_range},
            .r = 1.0f
        };

        // Top vertex
        d_coords_m_v(n_points_v + 3) = {
            .c = {avg[0], avg[1], avg[2] + 2*max_range},
            .r = 1.0f
        };
    });


    // First pass of step 1 (section 3.1 of the paper)
    // Count the number of neighbors by traversing the LBVH.
    //   if the sphere of radius 2alpha collides with an internal bounding box : go down the tree 
    //   if the node is a leaf, perform the neighbor predicate : ||s_i - s_j|| < 2min(alpha_i, alpha_j)
    //
    // Each time the neighbor predicate answers true, we increment the number of 
    // neighbor of this vertex.
    ava_for<256>(nullptr, 0, n_points, [=] __device__(uint32_t const pid) -> void {
        int stack[64];
        int stack_size      = 0;
        stack[stack_size++] = 0;
        uint32_t nneig_loc  = 0;

        Sphere3D const si = d_coords_m_v(pid);

        // 2*delta with delta = sqrt((1 + 5eps)/(1-3eps)) as described in
        // section 5.1 of the paper
        constexpr fp_tt const two_delta = 2.0000004768371866248429007339848490744624f;

        // Radius of the search sphere
        fp_tt const rs = two_delta*si.r;  

        while (stack_size != 0) {
            uint32_t const cur = stack[--stack_size];
            uint32_t const internal_sep = d_internal_sep_v(cur); 
            uint8_t const cl = d_child_is_leaf_v(cur);

            #pragma unroll 2
            for (uint32_t ichild = 0; ichild < 2; ichild++){
                bool intersect = true;
                if (!(cl & (ichild+1))) {
                    const BBox3D tmpbbox = d_bboxes_v(internal_sep+ichild);
                    // Perform a disk-bbox collision test
                    fp_tt const dmaxx = std::fmax(si.c[0] - tmpbbox.max(0), 0.0f);
                    fp_tt const dmaxy = std::fmax(si.c[1] - tmpbbox.max(1), 0.0f);
                    fp_tt const dmaxz = std::fmax(si.c[2] - tmpbbox.max(2), 0.0f);
                    fp_tt const dminx = std::fmax(tmpbbox.min(0) - si.c[0], 0.0f);
                    fp_tt const dminy = std::fmax(tmpbbox.min(1) - si.c[1], 0.0f);
                    fp_tt const dminz = std::fmax(tmpbbox.min(2) - si.c[2], 0.0f);
                    intersect = dmaxx*dmaxx + dminx*dminx + dmaxy*dmaxy + dminy*dminy + dmaxz*dmaxz + dminz*dminz < rs*rs; 
                    stack[stack_size] = internal_sep+ichild;
                    stack_size += intersect;
                } else {
                    Sphere3D const sj = d_coords_m_v(internal_sep+ichild);
                    // Check that || si - sj ||^2 < [2 min(alpha_i, alpha_j)]^2
                    fp_tt const min_alpha = std::fmin(si.r, sj.r);
                    intersect = (si.c - sj.c).sqnorm() < (two_delta*min_alpha)*(two_delta*min_alpha);
                    nneig_loc += intersect && (pid != internal_sep+ichild);
                }
            }
        }
        d_node_nineig_v(pid) = nneig_loc;
        d_row_offset_v(0) = 0;
    });

    // Compute the partial sum of the number of collisions
    ava::scan::inclusive_sum(
        nullptr,
        temp_mem_size, 
        d_node_nineig->data,
        d_row_offset->data + 1,
        n_points
    );
    temp_mem->resize({temp_mem_size});
    ava::scan::inclusive_sum(
        temp_mem->data,
        temp_mem_size,
        d_node_nineig->data,
        d_row_offset->data + 1,
        n_points
    );

    // Get the total number of collisions on host
    gpu_memcpy(&n_neig, d_row_offset->data + n_points, sizeof(n_neig), gpu_memcpy_device_to_host);

    AvaView<uint32_t, -1> d_offset_v = d_row_offset->to_view<-1>();
    AvaView<uint32_t, -1> d_block_offset_v = d_block_offset->to_view<-1>();
    uint32_t const c_n_points = n_points;

    // Compute the maximum number of collisions on each group of 32 nodes. 
    // This will allow us to compute the block offsets for the ELL/CSR format
    ava_for<256>(nullptr, 0, n_blocks, [=] __device__ (uint32_t const tid){
        uint32_t const start = tid*WARPSIZE;
        uint32_t const end = (tid+1)*WARPSIZE;
        uint32_t nnzmax_loc = 0; // N_INIT_TRI + 1;
        for (uint32_t i = start; i < (end < c_n_points ? end : c_n_points) ; i++){
            uint32_t const nnz = d_offset_v(i+1) - d_offset_v(i);
            if (nnz > nnzmax_loc) {
                nnzmax_loc = nnz;
            }
        }
        d_block_offset_v(tid+1) = 2*WARPSIZE*nnzmax_loc;
        d_block_offset_v(0) = 0;
    });

    ava::scan::inplace_inclusive_sum(
        nullptr,
        temp_mem_size,
        d_block_offset->data,
        n_blocks+1
    );
    temp_mem->resize({temp_mem_size});
    ava::scan::inplace_inclusive_sum(
        temp_mem->data,
        temp_mem_size, 
        d_block_offset->data,
        n_blocks+1
    );

    uint32_t total = 0;
    gpu_memcpy(&total, d_block_offset->data + n_blocks, sizeof(total), gpu_memcpy_device_to_host);

    // Allocate memory of ELL/CSR arrays
    d_node_triloc->resize({(int) (total + n_init_tri*n_points)});
    d_node_neig->resize({(int) total});
    d_active_neig->resize({(int) n_neig});
}

void AlphaShape3D::compute(){

    // ==================== Get the index of neighbors ==================

    const AvaView<int,     -1> d_internal_sep_v  = lbvh.d_internal_sep->to_view<-1>(); 
    const AvaView<uint8_t, -1> d_child_is_leaf_v = lbvh.d_child_is_leaf->to_view<-1>();
    const AvaView<BBox3D,  -1> d_bboxes_v        = lbvh.d_internal_data->to_view<-1>();  
    const AvaView<Sphere3D,   -1> d_coords_m_v      = lbvh.d_obj_m->to_view<-1>();
    const AvaView<uint32_t,  -1> d_node_nineig_v   = d_node_nineig->to_view<-1>();
    const TriLoc tinit = get_triloc_struct();

    // Second pass of step 1 (section 3.1 of the paper)
    // Fills the neighbors of each vertex by traversing the LBVH.
    //   if the sphere of radius 2alpha collides with an internal bounding box : go down the tree 
    //   if the node is a leaf, perform the neighbor predicate : ||s_i - s_j|| < 2min(alpha_i, alpha_j)
    //
    // Each time the neighbor predicate answers true, we add the corresponding neighbor 
    // in the array
    ava_for<256>(nullptr, 0, n_points, [=] __device__(uint32_t const tid) -> void {
            TriLoc tloc = tinit.thread_init(tid);
            int stack[64];
            int stack_size      = 0;
            stack[stack_size++] = 0;
            uint32_t nColl_loc   = n_inf_pts;

            Sphere3D const si = d_coords_m_v(tid);
            // 2*delta with delta = sqrt((1 + 4eps)/(1-3eps)) as described in
            // section 5.1 of the paper
            constexpr fp_tt const two_delta = 2.0000004768371866248429007339848490744624f;
            fp_tt const rs = two_delta*si.r;  // Radius of the search sphere : 

            while (stack_size != 0) {
                uint32_t const cur = stack[--stack_size];
                uint32_t const internal_sep = d_internal_sep_v(cur); 
                uint8_t const cl = d_child_is_leaf_v(cur);

                #pragma unroll 2
                for (uint32_t ichild = 0; ichild < 2; ichild++){
                    bool intersect = true;
                    if (!(cl & (ichild+1))) {
                        const BBox3D tmpbbox = d_bboxes_v(internal_sep+ichild);
                        // Perform a disk-bbox collision test
                        fp_tt const dmaxx = std::fmax(si.c[0] - tmpbbox.max(0), 0.0f);
                        fp_tt const dmaxy = std::fmax(si.c[1] - tmpbbox.max(1), 0.0f);
                        fp_tt const dmaxz = std::fmax(si.c[2] - tmpbbox.max(2), 0.0f);
                        fp_tt const dminx = std::fmax(tmpbbox.min(0) - si.c[0], 0.0f);
                        fp_tt const dminy = std::fmax(tmpbbox.min(1) - si.c[1], 0.0f);
                        fp_tt const dminz = std::fmax(tmpbbox.min(2) - si.c[2], 0.0f);
                        intersect = dmaxx*dmaxx + dminx*dminx + dmaxy*dmaxy + dminy*dminy + dmaxz*dmaxz + dminz*dminz < rs*rs; 
                        stack[stack_size] = internal_sep+ichild;
                        stack_size += intersect;
                    } else if (tid != internal_sep+ichild) {
                        Sphere3D const sj = d_coords_m_v(internal_sep+ichild);
                        // Check that || si - sj ||^2 < [2 min(alpha_i, alpha_j)]^2
                        fp_tt const min_alpha = std::fmin(si.r, sj.r);
                        intersect = (si.c - sj.c).sqnorm() < (two_delta*min_alpha)*(two_delta*min_alpha);
                        if (intersect){
                            tloc.set_neig(nColl_loc++) = internal_sep+ichild;
                        }
                    }
                }
            }
            if (nColl_loc > 128) printf("Particle %u has too much neighbors (%u > 128), only looking at first 128\n", tid, nColl_loc);
            d_node_nineig_v(tid) = nColl_loc > 128 ? 128 : nColl_loc;
        });

    gpu_memset(d_active_neig->data, 0, d_active_neig->size);

    AvaView<uint8_t, -1> d_active_neig_v = d_active_neig->to_view<-1>();
    AvaView<uint8_t, -1> d_node_is_bnd_v = d_node_is_bnd->to_view<-1>();
    AvaView<uint32_t, -1> d_trirow_v = d_trirow->to_view<-1>();
    AvaView<uint32_t, -1> d_row_v = d_row->to_view<-1>();
    AvaView<uint32_t, -1> d_node_nfneig_v = d_node_nfneig->to_view<-1>();
    AvaView<uint8_t, -1> d_node_ntri_v = d_node_ntri->to_view<-1>();
    AvaView<uint8_t, -1> d_node_ntri_out_v = d_node_ntri_out->to_view<-1>();
    AvaView<uint32_t, -1> d_row_offset_v = d_row_offset->to_view<-1>();
    
#define N_BND 128
    // Step 2 of the algorithm : section 3.2 of the paper
    ava_for<WARPSIZE>(nullptr, 0, n_points, [=, *this] __device__ (uint32_t const tid) {

        uint32_t const trank = tid % WARPSIZE;

        __shared__ uint8_t cycle[WARPSIZE * N_BND]; // allocate 128 bytes per thread for finding the boundary of the cavity

        // Init the local triangulation for this thread
        TriLoc tloc = tinit.thread_init(tid);

        // ================== Initialize Ti by inserting si ==================
        // bottom tet
        tloc.get_elem(0) = {0, 2, 1};

        // tet facing you
        tloc.get_elem(1) = {1, 2, 3};

        // left tet
        tloc.get_elem(2) = {0, 1, 3};

        // right tet
        tloc.get_elem(3) = {0, 3, 2};

        uint32_t const nColl_loc = d_node_nineig_v(tid);
        if (nColl_loc - n_inf_pts < dim) {
            d_node_is_bnd_v(tid) = true;
            d_node_ntri_out_v(tid) = 0;
            d_node_ntri_v(tid) = 0;
            d_node_nfneig_v(tid) = 0;
            return; // If less than dim neighbors, impossible to have a simplex (tri in 2D and tet in 3D)
        }
                                             
        uint32_t Tlast = n_init_tri;
        uint32_t ti;

        // ====================== Insertion loop ==========================
        for (uint32_t pi = n_inf_pts; pi < nColl_loc; ++pi) {

            // Get index of neighbor to insert in the current local triangulation
            uint32_t const curr_neig = tloc.get_neig(pi);

            // Clip vertices
            uint32_t Rsize = 0;
            ti             = 0;
            while (ti < Tlast) {

                // Get local triangle made by tid and two neighbors
                LocalElem const tri_loc = tloc.get_elem(ti);

                uint32_t i1 = tid;
                uint32_t i2 = tloc.get_neig(tri_loc.a);
                uint32_t i3 = tloc.get_neig(tri_loc.b);
                uint32_t i4 = tloc.get_neig(tri_loc.c);
                uint32_t i5 = curr_neig;

                Vec3f pa = d_coords_m_v(i1).c;
                Vec3f pb = d_coords_m_v(i2).c;
                Vec3f pc = d_coords_m_v(i3).c;
                Vec3f pd = d_coords_m_v(i4).c;
                Vec3f pe = d_coords_m_v(i5).c;

                // Inexact check
                fp_tt errbnd;
                fp_tt det = -insphere(&pa[0], &pb[0], &pc[0], &pd[0], &pe[0], &errbnd);

                // If error is too high, perform a simplified Simulation of Simplicity
                if (!((det > errbnd) || (-det > errbnd))){

                    // Order the vertices occording to their indices
                    bool swap = false;
                    uint32_t tmp;
                    Vec3f vtmp;
                    if ( i1 > i5 ) { tmp = i1; i1 = i5; i5 = tmp; vtmp = pa; pa = pe; pe = vtmp; swap=!swap; }
                    if ( i2 > i4 ) { tmp = i2; i2 = i4; i4 = tmp; vtmp = pb; pb = pd; pd = vtmp; swap=!swap; }
                    if ( i1 > i3 ) { tmp = i1; i1 = i3; i3 = tmp; vtmp = pa; pa = pc; pc = vtmp; swap=!swap; }
                    if ( i3 > i5 ) { tmp = i3; i3 = i5; i5 = tmp; vtmp = pc; pc = pe; pe = vtmp; swap=!swap; }
                    if ( i1 > i2 ) { tmp = i1; i1 = i2; i2 = tmp; vtmp = pa; pa = pb; pb = vtmp; swap=!swap; }
                    if ( i3 > i4 ) { tmp = i3; i3 = i4; i4 = tmp; vtmp = pc; pc = pd; pd = vtmp; swap=!swap; }
                    if ( i2 > i5 ) { tmp = i2; i2 = i5; i5 = tmp; vtmp = pb; pb = pe; pe = vtmp; swap=!swap; }
                    if ( i2 > i3 ) { tmp = i2; i2 = i3; i3 = tmp; vtmp = pb; pb = pc; pc = vtmp; swap=!swap; }
                    if ( i4 > i5 ) { tmp = i4; i4 = i5; i5 = tmp; vtmp = pd; pd = pe; pe = vtmp; swap=!swap; }
                    
                    // Compute exact predicate
                    det = -insphereexact(&pa[0], &pb[0], &pc[0], &pd[0], &pe[0]);
                    int depth = 0;

                    // While simulation of simplicity returns 0, evaluates minors of 
                    // the matrix
                    while (det == 0.0f){
                        depth++;
                        switch (depth) {
                            case 1: det = + orient3d(&pb[0], &pc[0], &pd[0], &pe[0]); break;
                            case 2: det = - orient3d(&pa[0], &pc[0], &pd[0], &pe[0]); break;
                            default: 
                                // The 5 points are coplanar, do not connect them
                                det = swap ? 1.0f : -1.0f;
                        }
                    }

                    if (swap) det = -det;
                }

                // If the inserted point is in the circumcircle of the triangle, 
                // add the triangle's edges to the cavity and remove it from the local 
                // triangulation
                if (det > 0.0f) {
                    Rsize++;

                    // Swap T[ti] with the last entry of the list
                    Tlast--;

                    tloc.get_elem(ti)    = tloc.get_elem(Tlast);
                    tloc.get_elem(Tlast) = tri_loc;
                } else {
                    ti++;
                }
            }

            // If the cavity is not empty : compute the Delaunay Ball Bij 
            // and add each new triangle in Ti
            if (Rsize){
                // Init the tri list
                for (uint32_t i = 0; i < N_BND; i++) cycle[WARPSIZE*i + trank] = 255;

                // Init the cycle with the last tri
                Rsize--;
                LocalElem const tri_last = tloc.get_elem(Tlast + Rsize);
                cycle[WARPSIZE*tri_last.a + trank] = tri_last.b;
                cycle[WARPSIZE*tri_last.b + trank] = tri_last.c;
                cycle[WARPSIZE*tri_last.c + trank] = tri_last.a;

                uint32_t cur = Tlast;
                uint32_t first_boundary = tri_last.a;

                int swap_with = (Tlast + Rsize) - 1;

                // Find the boundary of the cavity (topological circle)
                // Using a tailored version of Algorithm 4 presented in 
                // Ray, Nicolas, Dmitry Sokolov, Sylvain Lefebvre, and Bruno Lévy. 2018. “Meshless Voronoi on the GPU.” ACM Trans. Graph. 37 (6): 265:1-265:12. https://doi.org/10.1145/3272127.3275092.
                while (Rsize){
                    // Get a tet in the cavity
                    uint8_t tcur[dim] = {
                        tloc.get_elem(cur).a, 
                        tloc.get_elem(cur).b,
                        tloc.get_elem(cur).c
                    };

                    uint8_t tcycle[3] = {
                        cycle[WARPSIZE*tcur[0] + trank],
                        cycle[WARPSIZE*tcur[1] + trank],
                        cycle[WARPSIZE*tcur[2] + trank]
                    };

                    bool is_in_border[3] = {false, false, false};
                    bool next_is_opp[3] = {false, false, false};
                    for (uint32_t e = 0; e < 3; e++){
                        is_in_border[e] = (tcycle[e] != 255);
                        next_is_opp[e] = (tcycle[(e+1) % 3] == tcur[e]);
                    }

                    bool new_border_is_simple = (next_is_opp[0] || next_is_opp[1] || next_is_opp[2]);
                    for (uint32_t e = 0; e < 3; e++){
                        new_border_is_simple &= (next_is_opp[e] || next_is_opp[(e + 1) % 3] || !is_in_border[(e + 1) % 3]);
                    }

                    if (!new_border_is_simple){
                        cur++; 
                        if (cur > swap_with) cur = Tlast;
                        continue;
                    }

                    for (uint32_t e = 0; e < 3; e++){
                        if (!next_is_opp[e]){
                            cycle[WARPSIZE*tcur[e]+trank] = tcur[(e+1) % 3];
                        }
                    }

                    for (uint32_t e = 0; e < 3; e++){
                        if (next_is_opp[e] && next_is_opp[(e + 1) % 3]){
                            if (first_boundary == tcur[(e + 1) % 3]){
                                first_boundary = tcycle[(e + 1) % 3];
                            }
                            cycle[WARPSIZE*tcur[(e + 1) % 3]+trank] = 255;
                        }
                    }

                    tloc.get_elem(cur) = tloc.get_elem(swap_with);
                    tloc.get_elem(swap_with) = {tcur[0], tcur[1], tcur[2]};

                    swap_with--;
                    Rsize--;
                    cur = Tlast;
                }
                
                // Add one tet per edge of this circle
                cur = first_boundary;
                do {
                    tloc.get_elem(Tlast) = {(uint8_t) pi, (uint8_t) cur, cycle[WARPSIZE*cur + trank]};
                    Tlast++;
                    cur = cycle[WARPSIZE*cur+trank];
                } while (cur != first_boundary);
            }
        }


        // Only keep relevant vertices
        bool isBoundaryNode_loc = false;
        uint8_t nOutputTri_loc = 0;
        uint8_t nNodeTri_loc = 0;
        ti = 0;

        for (uint32_t i = 0; i < N_BND; i++) cycle[WARPSIZE*i + trank] = 0;

        while (ti < Tlast) {
            // Get local triangle
            LocalElem const tri_loc = tloc.get_elem(ti);

            // Vertices of the bounding boxes are always outside of acceptable zone
            if (tri_loc.a < n_inf_pts || tri_loc.b < n_inf_pts || tri_loc.c < n_inf_pts) {
                // If triangle is not accepted, remove it from the triangulation 
                // and flag the node as boundary
                isBoundaryNode_loc = true;
                Tlast--;
                tloc.get_elem(ti) = tloc.get_elem(Tlast);
                continue;
            } 

            uint32_t i1 = tid;
            uint32_t i2 = tloc.get_neig(tri_loc.a); 
            uint32_t i3 = tloc.get_neig(tri_loc.b);
            uint32_t i4 = tloc.get_neig(tri_loc.c);
            uint32_t tmp;
            if (i1 > i3) { tmp = i3; i3 = i1; i1 = tmp; }
            if (i2 > i4) { tmp = i4; i4 = i2; i2 = tmp; }
            if (i1 > i2) { tmp = i2; i2 = i1; i1 = tmp; }
            if (i3 > i4) { tmp = i4; i4 = i3; i3 = tmp; }
            if (i2 > i3) { tmp = i3; i3 = i2; i2 = tmp; }

            Sphere3D const pa = d_coords_m_v(i1);
            Sphere3D const pb = d_coords_m_v(i2);
            Sphere3D const pc = d_coords_m_v(i3);
            Sphere3D const pd = d_coords_m_v(i4);

            Vec3f const t = pa.c - pd.c;
            fp_tt const d2t = t.sqnorm();

            Vec3f const u = pb.c - pd.c;
            fp_tt const d2u = u.sqnorm();

            Vec3f const v = pc.c - pd.c;
            fp_tt const d2v = v.sqnorm();

            // The factor 1/6 is taken into account during division of circmuradius
            fp_tt vol = t[2] * (u[0]*v[1] - v[0]*u[1]) 
                      - u[2] * (t[0]*v[1] - v[0]*t[1]) 
                      + v[2] * (t[0]*u[1] - u[0]*t[1]);

            // |t|² u x v
            Vec3f uv = d2t * u.cross(v);

            // |u|² v x t
            uv += d2u * v.cross(t);

            // |v|² t x u
            uv += d2v * t.cross(u);

            fp_tt r_circ = (uv.sqnorm()) / (4.f * vol * vol);

            fp_tt const r1 = pa.r;
            fp_tt const r2 = pb.r;
            fp_tt const r3 = pc.r;
            fp_tt const r4 = pd.r;

            // If circumradius is accepted flag the neighbors as active, 
            // else remove the triangle and flag the node as boundary
            if (r_circ <= r1*r1 && r_circ <= r2*r2 && r_circ <= r3*r3 && r_circ <= r4*r4){
                d_active_neig_v(d_row_offset_v(tid) + tri_loc.a - n_inf_pts) += 1;
                d_active_neig_v(d_row_offset_v(tid) + tri_loc.b - n_inf_pts) += 1;
                d_active_neig_v(d_row_offset_v(tid) + tri_loc.c - n_inf_pts) += 1;
                // Count the number of neighbors
                cycle[WARPSIZE*tri_loc.a + trank]++;
                cycle[WARPSIZE*tri_loc.b + trank]++;
                cycle[WARPSIZE*tri_loc.c + trank]++;
                nNodeTri_loc++;

                // Output only one tet for a group of 4 points
                if (tid == i1) {
                    nOutputTri_loc++;
                    ti++;
                } else {
                    Tlast--;
                    tloc.get_elem(ti) = tloc.get_elem(Tlast);
                    tloc.get_elem(Tlast) = tri_loc;
                }
            } else {
                isBoundaryNode_loc = true;
                Tlast--;
                tloc.get_elem(ti) = tloc.get_elem(Tlast);
            }
        }

        d_node_is_bnd_v(tid) = isBoundaryNode_loc;
        d_node_ntri_out_v(tid) = nOutputTri_loc;
        d_node_ntri_v(tid) = nNodeTri_loc;

        uint32_t d_node_nfneig_loc = 0;
        for (uint32_t i = 0; i < N_BND; i++) d_node_nfneig_loc += (cycle[WARPSIZE*i + trank] != 0);
        d_node_nfneig_v(tid) = d_node_nfneig_loc;

        d_row_v(0) = 0;
        d_trirow_v(0) = 0;
    });


    // ==================== Scan the number of edges/tri per nodes ============
    ava::scan::inclusive_sum(nullptr, temp_mem_size, d_node_ntri_out->data, d_trirow->data + 1, n_points);
    temp_mem->resize({temp_mem_size});

    ava::scan::inclusive_sum(temp_mem->data, temp_mem_size, d_node_ntri_out->data, d_trirow->data + 1, n_points);
    gpu_memcpy(&n_tri, d_trirow->data + n_points, sizeof(n_tri), gpu_memcpy_device_to_host);

    ava::scan::inclusive_sum(temp_mem->data, temp_mem_size, d_node_nfneig->data, d_row->data + 1, n_points);
    gpu_memcpy(&n_edges, d_row->data + n_points, sizeof(n_edges), gpu_memcpy_device_to_host);
}

void AlphaShape3D::compress() {
    d_neig->resize({(int) n_edges});
    d_edge_is_bnd->resize({(int) n_edges});
    d_triglob->resize({(int) n_tri});

    AvaView<uint32_t, -1> d_node_nineig_v = d_node_nineig->to_view<-1>();
    AvaView<uint8_t, -1> d_active_neig_v = d_active_neig->to_view<-1>();
    AvaView<uint8_t, -1> d_edge_is_bnd_v = d_edge_is_bnd->to_view<-1>();
    AvaView<uint32_t, -1> d_neig_v = d_neig->to_view<-1>();
    AvaView<Elem, -1> d_triglob_v = d_triglob->to_view<-1>();
    AvaView<uint32_t, -1> d_trirow_v = d_trirow->to_view<-1>();
    AvaView<uint32_t, -1> d_row_v = d_row->to_view<-1>();
    AvaView<uint32_t, -1> d_row_offset_v = d_row_offset->to_view<-1>();
    TriLoc const tinit = get_triloc_struct();

    ava_for<256>(nullptr, 0, n_points, [=, *this] __device__ (uint32_t const tid) {
        TriLoc tloc = tinit.thread_init(tid);
        uint32_t put_idx = d_row_v(tid);
        for (uint32_t i = n_inf_pts; i < d_node_nineig_v(tid); ++i) {
            uint8_t const edge_count = d_active_neig_v(d_row_offset_v(tid) + i - n_inf_pts);
            if (edge_count) {
                d_edge_is_bnd_v(put_idx) = (edge_count <= 3);
                d_neig_v(put_idx++) = tloc.get_neig(i);
            }
        }

        uint32_t triIdx = 0;
        for (uint32_t i = d_trirow_v(tid); i < d_trirow_v(tid+1); i++){
            d_triglob_v(i).a = tid;
            LocalElem const triloc = tloc.get_elem(triIdx);
            d_triglob_v(i).b = tloc.get_neig(triloc.a);
            d_triglob_v(i).c = tloc.get_neig(triloc.b);
            d_triglob_v(i).d = tloc.get_neig(triloc.c);
            triIdx++;
        }
    });
}

} // namespace stream::mesh
