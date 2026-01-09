#include <functional>
#include <cstdio>

#include "defines.h"
#include "DirectAlphaShape2D.hpp"
#include "ava.h"
#include "ava_host_array.hpp"
#include "ava_scan.h"
#include "ava_view.h"
#include "primitives.hpp"
#include "predicates.hpp"
#include "lbvh.hpp"

namespace stream::mesh {

AlphaShape2D::AlphaShape2D() {
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

void AlphaShape2D::set_nodes(const AvaHostArray<Sphere2D, int>::Ptr h_nodes) {
    n_points = h_nodes->size();
    d_coords = AvaDeviceArray<Sphere2D, int>::create({(int) n_points});
    d_coords->set(h_nodes);
}

uint32_t AlphaShape2D::getPermutation(std::vector<uint32_t>& perm) const{
    perm.resize(n_points);
    gpu_memcpy(perm.data(), lbvh.d_map_sorted->data, sizeof(perm[0])*n_points, gpu_memcpy_device_to_host);

    return perm.size();
}

uint32_t AlphaShape2D::getTri(std::vector<Elem>& tri) const {
    tri.resize(d_triglob->size);
    gpu_memcpy(tri.data(), d_triglob->data, sizeof(tri[0])*d_triglob->size, gpu_memcpy_device_to_host);
    return tri.size();
}

uint32_t AlphaShape2D::getEdge(std::vector<uint32_t>& nEdgeNodes, std::vector<uint32_t>& edges) const {
    nEdgeNodes.resize(n_points+1);
    edges.resize(n_edges);

    gpu_memcpy(nEdgeNodes.data(), d_row->data, sizeof(nEdgeNodes[0])*(n_points+1), gpu_memcpy_device_to_host);
    gpu_memcpy(edges.data(), d_neig->data, sizeof(edges[0])*(n_edges), gpu_memcpy_device_to_host);

    return edges.size();
}

uint32_t AlphaShape2D::getBoundaryEdges(std::vector<uint8_t>& edge_is_bnd) const {
    edge_is_bnd.resize(n_edges);
    gpu_memcpy(edge_is_bnd.data(), d_edge_is_bnd->data, sizeof(edge_is_bnd[0])*(n_edges), gpu_memcpy_device_to_host);
    return edge_is_bnd.size();
}

uint32_t AlphaShape2D::getBoundaryNodes(std::vector<uint8_t>& node_is_bnd) const {
    node_is_bnd.resize(n_points);
    gpu_memcpy(node_is_bnd.data(), d_node_is_bnd->data, sizeof(node_is_bnd[0])*(n_points), gpu_memcpy_device_to_host);
    return node_is_bnd.size();
}

uint32_t AlphaShape2D::getCoordsMorton(std::vector<Sphere2D>& coords_m) const {
    coords_m.resize(n_points);
    gpu_memcpy(coords_m.data(), lbvh.d_obj_m->data, sizeof(coords_m[0])*(n_points), gpu_memcpy_device_to_host);
    return coords_m.size();
}


void AlphaShape2D::init() {

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

    lbvh.set_objects(d_coords);
    lbvh.build();

    // Set infinity points at the end of the morton-reordered nodes
    int n_points_v = n_points;
    lbvh.d_obj_m->resize({(int) (n_points+n_inf_pts)});

    const AvaView<Sphere2D, -1> d_coords_m_v = lbvh.d_obj_m->to_view<-1>();
    BBoxT const * const __restrict__ d_bb_glob = lbvh.d_bb_glob->data;
    ava_for<1>(nullptr, 0, 1, [=] __device__ (int const tid) {
        VecT const avg = 0.5f*(d_bb_glob->pmax + d_bb_glob->pmin);
        VecT const range = d_bb_glob->pmax - d_bb_glob->pmin;
        fp_tt const max_range   = fmaxf(range[0], range[1]);
        // Bottom left
        d_coords_m_v(n_points_v) = {
            .c = {avg[0] - 2*max_range, avg[1] - 2*max_range}, 
            .r = 0.0f
        };
        // Bottom right
        d_coords_m_v(n_points_v+1) = {
            .c = {avg[0] + 2*max_range, avg[1] - 2*max_range}, 
            .r = 0.0f
        };
        // Top
        d_coords_m_v(n_points_v+2) = {
            .c = {avg[0], avg[1] + 2*max_range},
            .r = 0.0f
        };
    });

    const AvaView<int,     -1> d_internal_sep_v  = lbvh.d_internal_sep->to_view<-1>(); 
    const AvaView<uint8_t, -1> d_child_is_leaf_v = lbvh.d_child_is_leaf->to_view<-1>();
    const AvaView<BBox2D,  -1> d_bboxes_v        = lbvh.d_internal_data->to_view<-1>();
    const AvaView<uint32_t,  -1> d_node_nineig_v   = d_node_nineig->to_view<-1>();
    const AvaView<uint32_t,  -1> d_row_offset_v   = d_row_offset->to_view<-1>();

    // First pass of step 1 (section 3.1 of the paper)
    // Count the number of neighbors by traversing the LBVH.
    //   if the sphere of radius 2alpha collides with an internal bounding box : go down the tree 
    //   if the node is a leaf, perform the neighbor predicate : ||s_i - s_j|| < 2min(alpha_i, alpha_j)
    //
    // Each time the neighbor predicate answers true, we increment the number of 
    // neighbor of this vertex.
    ava_for<256>(nullptr, 0, n_points, [=] __device__(uint32_t const tid) -> void {
        int stack[64];
        int stack_size      = 0;
        stack[stack_size++] = 0;
        uint32_t nneig_loc  = 0;

        Sphere2D const si = d_coords_m_v(tid);

        // 2*delta with delta = sqrt((1 + 4eps)/(1-3eps)) as described in
        // section 5.1 of the paper
        constexpr fp_tt const two_delta = 2.0000004172325445139859201780136934301654f;

        // Radius of the search sphere
        fp_tt const rs = two_delta*si.r;  

        // Bounding box of the searched sphere
        // fp_tt const pmaxx = si.c[0] + rs;
        // fp_tt const pmaxy = si.c[1] + rs;
        // fp_tt const pminx = si.c[0] - rs;
        // fp_tt const pminy = si.c[1] - rs;

        while (stack_size != 0) {
            uint32_t const cur = stack[--stack_size];
            uint32_t const internal_sep = d_internal_sep_v(cur); 
            uint8_t const cl = d_child_is_leaf_v(cur); // 0-th bit : left child 
                                                       // 1-st bit : right child

            #pragma unroll 2
            for (uint32_t ichild = 0; ichild < 2; ichild++){
                bool intersect = true;
                if (!(cl & (ichild+1))) {
                    const BBox2D tmpbbox = d_bboxes_v(internal_sep+ichild);
                    // intersect &= pmaxx > tmpbbox.min(0);
                    // intersect &= pmaxy > tmpbbox.min(1);
                    // intersect &= pminx < tmpbbox.max(0);
                    // intersect &= pminy < tmpbbox.max(1);
                    fp_tt const dmaxx = std::fmax(si.c[0] - tmpbbox.max(0), 0.0f);
                    fp_tt const dmaxy = std::fmax(si.c[1] - tmpbbox.max(1), 0.0f);
                    fp_tt const dminx = std::fmax(tmpbbox.min(0) - si.c[0], 0.0f);
                    fp_tt const dminy = std::fmax(tmpbbox.min(1) - si.c[1], 0.0f);
                    intersect = dmaxx*dmaxx + dminx*dminx + dmaxy*dmaxy + dminy*dminy < rs*rs; 
                    stack[stack_size] = internal_sep+ichild;
                    stack_size += intersect;
                } else {
                    Sphere2D const sj = d_coords_m_v(internal_sep+ichild);
                    // Check that || si - sj ||^2 < [2 min(alpha_i, alpha_j)]^2
                    fp_tt const min_alpha = std::fmin(si.r, sj.r);
                    intersect = (si.c - sj.c).sqnorm() < (two_delta*min_alpha)*(two_delta*min_alpha);

                    // Avoid self-collisions
                    nneig_loc += intersect && (tid != internal_sep+ichild);
                }
            }
        }
        d_node_nineig_v(tid) = nneig_loc;
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

    
    // Compute the maximum number of collisions on each group of 32 nodes. 
    // This will allow us to compute the block offsets for the ELL/CSR format
    AvaView<uint32_t, -1> d_block_offset_v = d_block_offset->to_view<-1>();
    uint32_t const c_n_points = n_points;
    ava_for<256>(nullptr, 0, n_blocks, [=] __device__ (uint32_t const tid){
        uint32_t const start = tid*WARPSIZE;
        uint32_t const end = (tid+1)*WARPSIZE;
        uint32_t nnzmax_loc = 0; // N_INIT_TRI + 1;
        for (uint32_t i = start; i < (end < c_n_points ? end : c_n_points) ; i++){
            uint32_t const nnz = d_row_offset_v(i+1) - d_row_offset_v(i);
            if (nnz > nnzmax_loc) {
                nnzmax_loc = nnz;
            }
        }
        d_block_offset_v(tid+1) = WARPSIZE*nnzmax_loc;
        d_block_offset_v(0) = 0;
    });

    // Perform the partial sum of the blocks
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
    d_active_neig->resize({(int)n_neig});
}

void AlphaShape2D::compute(){

    // ==================== Get the index of collisions ==================
    const AvaView<int,     -1> d_internal_sep_v  = lbvh.d_internal_sep->to_view<-1>(); 
    const AvaView<uint8_t, -1> d_child_is_leaf_v = lbvh.d_child_is_leaf->to_view<-1>();
    const AvaView<BBox2D,  -1> d_bboxes_v        = lbvh.d_internal_data->to_view<-1>();  
    const AvaView<Sphere2D,-1> d_coords_m_v      = lbvh.d_obj_m->to_view<-1>();
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
            uint32_t nneig_loc = n_inf_pts;
            int stack[64];
            int stack_size      = 0;
            stack[stack_size++] = 0;

            Sphere2D const si = d_coords_m_v(tid);
            // 2*delta with delta = sqrt((1 + 4eps)/(1-3eps)) as described in
            // section 5.1 of the paper
            constexpr fp_tt const two_delta = 2.0000004172325445139859201780136934301654f;
            fp_tt const rs = two_delta*si.r;  // Radius of the search sphere : 

            while (stack_size != 0) {
                uint32_t const cur = stack[--stack_size];
                uint32_t const internal_sep = d_internal_sep_v(cur); 
                uint8_t const cl = d_child_is_leaf_v(cur);

                #pragma unroll 2
                for (uint32_t ichild = 0; ichild < 2; ichild++){
                    bool intersect = true;
                    if (!(cl & (ichild+1))) {
                        const BBox2D tmpbbox = d_bboxes_v(internal_sep+ichild);
                        // Perform a sphere-BBox intersection test
                        fp_tt const dmaxx = std::fmax(si.c[0] - tmpbbox.max(0), 0.0f);
                        fp_tt const dmaxy = std::fmax(si.c[1] - tmpbbox.max(1), 0.0f);
                        fp_tt const dminx = std::fmax(tmpbbox.min(0) - si.c[0], 0.0f);
                        fp_tt const dminy = std::fmax(tmpbbox.min(1) - si.c[1], 0.0f);
                        intersect = dmaxx*dmaxx + dminx*dminx + dmaxy*dmaxy + dminy*dminy < rs*rs; 
                        stack[stack_size] = internal_sep+ichild;
                        stack_size += intersect;
                    } else if (tid != internal_sep+ichild) {
                        Sphere2D const sj = d_coords_m_v(internal_sep+ichild);
                        // Check that || si - sj ||^2 < [2 min(alpha_i, alpha_j)]^2
                        fp_tt const min_alpha = std::fmin(si.r, sj.r);
                        intersect = (si.c - sj.c).sqnorm() < (two_delta*min_alpha)*(two_delta*min_alpha);
                        if (intersect){
                            tloc.set_neig(nneig_loc++) = internal_sep+ichild;
                        }
                    }
                }
                if (nneig_loc > 250) {
                    printf("Particle %u has too much neighbors (%u > 250), only looking at first 250\n", tid, nneig_loc);
                    break;
                }
            }
            d_node_nineig_v(tid) = nneig_loc;
        });


    // Reset d_active_neig
    gpu_memset(d_active_neig->data, 0, d_active_neig->size);

    AvaView<uint8_t, -1> d_node_is_bnd_v    = d_node_is_bnd->to_view<-1>(); 
    AvaView<uint32_t,  -1> d_node_nfneig_v   = d_node_nfneig->to_view<-1>(); 
    AvaView<uint8_t, -1> d_node_ntri_v     = d_node_ntri->to_view<-1>();
    AvaView<uint8_t, -1> d_node_ntri_out_v = d_node_ntri_out->to_view<-1>();
    AvaView<uint8_t, -1> d_active_neig_v    = d_active_neig->to_view<-1>();
    AvaView<uint32_t,  -1> d_row_offset_v    = d_row_offset->to_view<-1>();   
    AvaView<uint32_t,  -1> d_row_v           = d_row->to_view<-1>();  
    AvaView<uint32_t,  -1> d_trirow_v        = d_trirow->to_view<-1>();
    
    // Step 2 of the algorithm : section 3.2 of the paper
    ava_for<WARPSIZE>(nullptr, 0, n_points, [=] __device__ (uint32_t const si) {

        // Init the local triangulation for this thread
        TriLoc tloc = tinit.thread_init(si);

        // ================== Initialize Ti by inserting si ==================
        // Top left triangle
        tloc.get_elem(0) = {1, 2};

        // Top right
        tloc.get_elem(1) = {2, 0};

        // Bottom right
        tloc.get_elem(2) = {0, 1};


        uint32_t const nneig_loc = d_node_nineig_v(si);
        if (nneig_loc - n_inf_pts < dim) {
            d_node_is_bnd_v(si) = true;
            d_node_ntri_out_v(si) = 0;
            d_node_ntri_v(si) = 0;
            d_node_nfneig_v(si) = 0;
            return; // If less than dim neighbors, impossible to have a simplex (tri in 2D and tet in 3D)
        }
                                             
        // 256 bitset : i-th bit indicates if i-th local node is used
        __shared__ uint64_t neig_tag_loc[WARPSIZE*4];
        uint32_t const trank = si % WARPSIZE;

        uint32_t Tlast = n_init_tri;
        uint32_t ti; // current element index in Ti

        // ====================== Insertion loop ==========================
        for (uint32_t ni = n_inf_pts; ni < nneig_loc; ++ni) {

            // Reset Delaunay Cavity Cij
            neig_tag_loc[WARPSIZE*0 + trank] = UINT64_C(0);
            neig_tag_loc[WARPSIZE*1 + trank] = UINT64_C(0);
            neig_tag_loc[WARPSIZE*2 + trank] = UINT64_C(0);
            neig_tag_loc[WARPSIZE*3 + trank] = UINT64_C(0);

            // Get index of neighbor to insert in the current local triangulation
            uint32_t const sj = tloc.get_neig(ni);

            // Compute the Delaunay Cavity Cij
            uint32_t Rsize = 0;
            ti            = 0;
            while (ti < Tlast) {
                // Get local triangle made by si and its two neighbors
                LocalElem const tri_loc = tloc.get_elem(ti);

                // Indices of vertices in global array
                uint32_t i1 = si;
                uint32_t i2 = tloc.get_neig(tri_loc.a);
                uint32_t i3 = tloc.get_neig(tri_loc.b);
                uint32_t i4 = sj;

                // Coordinates of vertices
                Vec2f pa = d_coords_m_v(i1).c;
                Vec2f pb = d_coords_m_v(i2).c;
                Vec2f pc = d_coords_m_v(i3).c;
                Vec2f pd = d_coords_m_v(i4).c;

                // Vec2f const ac = pa - pc;
                // Vec2f const bc = pb - pc;
                // fp_tt const orientation = ac[0]*bc[1] - ac[1]*bc[0];
                fp_tt const orientation = orient2d(&pa[0], &pb[0], &pc[0]);

                // Inexact check
                fp_tt errbound;
                fp_tt det = incircle(&pa[0], &pb[0], &pc[0], &pd[0], &errbound);

                // If error is too high, perform a simplified Simulation of Simplicity
                if (!((det > errbound) || (-det > errbound))) {

                    // Order the vertices occording to their indices
                    bool swap = false;
                    uint32_t tmp;
                    Vec2f vtmp;
                    if (i1 > i3) { tmp = i3; i3 = i1; i1 = tmp; vtmp = pc; pc = pa; pa = vtmp; swap = !swap; }
                    if (i2 > i4) { tmp = i4; i4 = i2; i2 = tmp; vtmp = pd; pd = pb; pb = vtmp; swap = !swap; }
                    if (i1 > i2) { tmp = i2; i2 = i1; i1 = tmp; vtmp = pb; pb = pa; pa = vtmp; swap = !swap; }
                    if (i3 > i4) { tmp = i4; i4 = i3; i3 = tmp; vtmp = pd; pd = pc; pc = vtmp; swap = !swap; }
                    if (i2 > i3) { tmp = i3; i3 = i2; i2 = tmp; vtmp = pc; pc = pb; pb = vtmp; swap = !swap; }

                    // Compute exact predicate
                    det = incircleexact(&pa[0], &pb[0], &pc[0], &pd[0]);

                    // While simulation of simplicity returns 0, evaluates minors of 
                    // the matrix
                    int depth = 0;
                    while (det == 0.0f){
                        depth++;
                        switch (depth){
                            case 1: det = + orient2d(&pb[0], &pc[0], &pd[0]); break;
                            case 2: det = - orient2d(&pa[0], &pc[0], &pd[0]); break;
                            default:
                                // The 4 points are collinear, do not connect them
                                det = (swap + (orientation < 0.0f)) == 1 ? 1.0f : -1.0f;
                                break;
                        }
                    }

                    if (swap) det = -det;
                }

                // Account for the fact that the triangle may not be correctly 
                // oriented
                if (orientation < 0.0f) det = -det;

                // If the inserted point is in the circumcircle of the triangle, 
                // add the triangle to the cavity Cij
                if (det > 0.0f) {
                    // Add T[ti] to the used edges
                    neig_tag_loc[WARPSIZE*(tri_loc.a >> 6ULL) + trank] ^= 1ULL << (tri_loc.a & 63ULL);
                    neig_tag_loc[WARPSIZE*(tri_loc.b >> 6ULL) + trank] ^= 1ULL << (tri_loc.b & 63ULL);
                    Rsize++;

                    // Remove T[ti] from T
                    Tlast--;
                    tloc.get_elem(ti) = tloc.get_elem(Tlast);
                } else {
                    ti++;
                }
            }

            // If the cavity is not empty : compute the Delaunay Ball Bij 
            // and add each new triangle in Ti
            if (Rsize){
                // Add an outer loop to avoid indexing neig_tag_loc (shared memory) 
                // in the inner loop
                for (uint32_t j = 0; j < 4; j++){
                    uint64_t const v = neig_tag_loc[WARPSIZE*j+trank];
                    uint32_t const end = (j+1)*64 > ni ? ni : (j+1)*64;
                    if (!v) continue; 
                                      
                    for (uint32_t r = j*64; r < end; ++r) {
                        if (v & (1ULL << (r & 63ULL)) ) {
                            tloc.get_elem(Tlast) = {(uint8_t) r, (uint8_t) ni};
                            Tlast++;
                        }
                    }
                }
            }
        }

        neig_tag_loc[WARPSIZE*0 + trank] = UINT64_C(0);
        neig_tag_loc[WARPSIZE*1 + trank] = UINT64_C(0);
        neig_tag_loc[WARPSIZE*2 + trank] = UINT64_C(0);
        neig_tag_loc[WARPSIZE*3 + trank] = UINT64_C(0);

        // Only keep relevant vertices
        bool node_is_bnd_loc = false;
        uint8_t ntri_out_loc = 0;
        uint8_t ntri_loc = 0;
        ti = 0;

        // Step 3 : filtering (section 3.3 of the paper)
        while (ti < Tlast) {
            LocalElem const tri_loc = tloc.get_elem(ti);

            // Delete the element if it contains infinity vertices
            if (tri_loc.a < n_inf_pts || tri_loc.b < n_inf_pts) {
                node_is_bnd_loc = true; // flag the current node as boundary
                Tlast--;
                tloc.get_elem(ti) = tloc.get_elem(Tlast);
                continue;
            }

            // Sort the vertices according to their indices as described 
            // in section 5.3 of the paper
            uint32_t i1 = si;
            uint32_t i2 = tloc.get_neig(tri_loc.a);
            uint32_t i3 = tloc.get_neig(tri_loc.b);
            uint32_t tmp;
            bool swap = false;
            if (i3 < i2) {tmp = i3; i3 = i2; i2 = tmp; swap = !swap;}
            if (i2 < i1) {tmp = i2; i2 = i1; i1 = tmp; swap = !swap;}
            if (i3 < i2) {tmp = i3; i3 = i2; i2 = tmp; swap = !swap;}

            // Compute circumradius of local triangle as the
            // product of edge length divided by 4 times the area
            Sphere2D const p1 = d_coords_m_v(i1);
            Sphere2D const p2 = d_coords_m_v(i2);
            Sphere2D const p3 = d_coords_m_v(i3);

            fp_tt det = (p3.c - p2.c).cross(p1.c - p2.c);
            if (swap) det = -det;
            fp_tt circum_rsqr = (
                ( (p1.c - p2.c).sqnorm() * (p2.c - p3.c).sqnorm() * (p3.c - p1.c).sqnorm()) / ( 4.f*det*det)
            );
            // If circumradius is accepted flag the neighbors as active, 
            // else remove the triangle and flag the node as boundary
            // if (circum_rsqr <= p1.r*p1.r && circum_rsqr <= p2.r*p2.r && circum_rsqr <= p3.r*p3.r) {
            fp_tt const a_eff = std::fmin(p3.r, std::fmin(p1.r, p2.r));
            if (circum_rsqr <= a_eff*a_eff) {
                neig_tag_loc[WARPSIZE*(tri_loc.a >> 6ULL) + trank] |= 1ULL << (tri_loc.a & 63ULL);
                neig_tag_loc[WARPSIZE*(tri_loc.b >> 6ULL) + trank] |= 1ULL << (tri_loc.b & 63ULL);
                d_active_neig_v(d_row_offset_v(si) + tri_loc.a - n_inf_pts) += 1;
                d_active_neig_v(d_row_offset_v(si) + tri_loc.b - n_inf_pts) += 1;
                ntri_loc++;

                // The triangle is inverted : reverse the point ordering
                if (det < 0){
                    tloc.get_elem(ti) = {tri_loc.b, tri_loc.a};
                }
                ti++;
                ntri_out_loc += (si == i1); // Make sure to count the triangle only once
            } else {
                // Remove T[ti] from T if the triangle is not accepted
                node_is_bnd_loc = true;
                Tlast--;
                tloc.get_elem(ti) = tloc.get_elem(Tlast);
            }
        }

        d_node_is_bnd_v(si)   = node_is_bnd_loc;
        d_node_ntri_out_v(si) = ntri_out_loc;
        d_node_ntri_v(si)     = ntri_loc;
        // Count number of bits set to 1 (i.e. the number of final neighbors)
        d_node_nfneig_v(si) = __builtin_popcountll(neig_tag_loc[WARPSIZE*0+trank]) 
                            + __builtin_popcountll(neig_tag_loc[WARPSIZE*1+trank])
                            + __builtin_popcountll(neig_tag_loc[WARPSIZE*2+trank]) 
                            + __builtin_popcountll(neig_tag_loc[WARPSIZE*3+trank]);

        if (si == 0) {
            d_trirow_v(0) = 0;
            d_row_v(0) = 0;
        }
    });

    // ==================== Scan the number of edges/tri per nodes ============
    
    ava::scan::inclusive_sum(
        nullptr,
        temp_mem_size,
        d_node_ntri_out->data,
        d_trirow->data + 1,
        n_points
    );
    temp_mem->resize({temp_mem_size});
    ava::scan::inclusive_sum(
        temp_mem->data,
        temp_mem_size, 
        d_node_ntri_out->data, 
        d_trirow->data + 1,
        n_points
    );
    gpu_memcpy(&n_tri, d_trirow->data + n_points, sizeof(n_tri), gpu_memcpy_device_to_host);

    ava::scan::inclusive_sum(
        nullptr,
        temp_mem_size,
        d_node_nfneig->data,
        d_row->data + 1,
        n_points
    );
    temp_mem->resize({temp_mem_size});
    ava::scan::inclusive_sum(
        temp_mem->data,
        temp_mem_size,
        d_node_nfneig->data,
        d_row->data + 1,
        n_points
    );
    gpu_memcpy(&n_edges, d_row->data + n_points, sizeof(n_edges), gpu_memcpy_device_to_host);
}

void AlphaShape2D::compress() {
    d_neig->resize({(int) n_edges});
    d_edge_is_bnd->resize({(int) n_edges});
    d_triglob->resize({(int) n_tri});

    AvaView<uint8_t, -1> d_edge_is_bnd_v = d_edge_is_bnd->to_view<-1>(); 
    AvaView<uint32_t,  -1> d_node_nineig_v = d_node_nineig->to_view<-1>();
    AvaView<uint32_t,  -1> d_neig_v        = d_neig->to_view<-1>(); 
    AvaView<uint8_t, -1> d_active_neig_v = d_active_neig->to_view<-1>();
    AvaView<uint32_t,  -1> d_row_offset_v = d_row_offset->to_view<-1>();   
    AvaView<uint32_t,  -1> d_row_v        = d_row->to_view<-1>();  
    AvaView<uint32_t,  -1> d_trirow_v     = d_trirow->to_view<-1>();
    AvaView<Elem,   -1> d_triglob_v     = d_triglob->to_view<-1>();
    TriLoc const tinit = get_triloc_struct();

    ava_for<256>(nullptr, 0, n_points, [=] __device__(uint32_t const si) {
        TriLoc tloc = tinit.thread_init(si);
        uint32_t put_idx = d_row_v(si);
        for (uint32_t i = n_inf_pts; i < d_node_nineig_v(si); ++i) {
            uint8_t const edge_count = d_active_neig_v(d_row_offset_v(si) + i - n_inf_pts);
            if (edge_count) {
                d_edge_is_bnd_v(put_idx) = (edge_count == 1);
                d_neig_v(put_idx++) = tloc.get_neig(i);
            }
        }

        uint32_t triIdx = 0;
        for (uint32_t i = d_trirow_v(si); i < d_trirow_v(si+1); i++){
            d_triglob_v(i).a = si;
            LocalElem tricur = tloc.get_elem(triIdx);
            uint32_t neig[2] = {
                tloc.get_neig(tricur.a), 
                tloc.get_neig(tricur.b)
            };

            // Skip the triangles that are output by another node
            while (neig[0] < si || neig[1] < si) {
                triIdx++;
                tricur  = tloc.get_elem(triIdx);
                neig[0] = tloc.get_neig(tricur.a); 
                neig[1] = tloc.get_neig(tricur.b);
            }
            d_triglob_v(i).b = neig[0];
            d_triglob_v(i).c = neig[1];
            triIdx++;
        }
    });
}

} // namespace stream::mesh
