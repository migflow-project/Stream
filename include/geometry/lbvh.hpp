#ifndef __STREAM_LBVH_HPP__
#define __STREAM_LBVH_HPP__

// Forward declarations
namespace stream::geo {
    template<typename ObjT, typename DataF, int dim> struct LBVH;
} // namespace stream::geo

// Typedefs for often used template arguments
#ifdef __cplusplus
extern "C" {
#endif
    
#ifdef __cplusplus
}
#endif

namespace stream::geo {

    template<typename ObjT, typename DataF, int dim>
    struct LBVH {

        // Compute the bounding box of the set of objects
        void _compute_global_bbox();

        // Compute the morton codes of the centroid of each object
        void _compute_morton_codes();

        // Sort the objects according to their morton code
        void _sort_objects();

        // Build the hierachy of the LBVH
        void _build_hierarchy();

        // Initialize data
        void init();

        // Build the LBVH tree
        void build();

        // Fit internal node data
        void fit();
    };

} // namespace stream::geo

#endif // __STREAM_LBVH_HPP__

