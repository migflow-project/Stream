#ifndef __STREAM_TRAVERSAL_STACK_HPP__
#define __STREAM_TRAVERSAL_STACK_HPP__

#include <utility>
#include "defines.h"

// forward declarations 
namespace stream::geo {

    // Define a min heap for informed traversal of BVH
    // pop() always returns the item in the stack with the smallest cost
    template<typename IdxT, typename CostT, int size> 
    struct TraversalMinHeap {
        static constexpr int MaxSize = size;

        struct Pair {
            IdxT first;
            CostT second;
        };

        int len = 0;
        Pair stack[size] = {};

        TraversalMinHeap() = default;
        ~TraversalMinHeap() = default;

        __host__ __device__ inline Pair peek() const {
            return stack[0];
        }

        __host__ __device__ void inline push(IdxT idx, CostT cost) {
            stack[len++] = {idx, cost};

            int index = len - 1;
            int parent = (index - 1) >> 1;
            while (index > 0 && stack[parent].second > stack[index].second) {
                Pair tmp = stack[index];
                stack[index] = stack[parent];
                stack[parent] = tmp;
                // Move up the tree to the
                //parent of the current element
                index = parent;
                parent = (index - 1) >> 1;
            }
        }

        __host__ __device__ inline void pop() { 
            stack[0] = stack[--len];

            // Heapify the tree starting from the element at the
            // deleted index
            int index = 0;
            while (true) {
                int left_child = 2 * index + 1;
                int right_child = 2 * index + 2;
                int smallest = index;

                if (left_child < len && stack[left_child].second < stack[smallest].second) {
                    smallest = left_child;
                }
                if (right_child < len && stack[right_child].second < stack[smallest].second) {
                    smallest = right_child;
                }
                if (smallest != index) {
                    Pair tmp = stack[index];
                    stack[index] = stack[smallest];
                    stack[smallest] = tmp;
                    index = smallest;
                } else {
                    break;
                }
            }
        }
    };

    // Define a min heap for informed traversal of BVH
    // pop() always returns the item in the stack with the smallest cost
    template<typename IdxT, int size> 
    struct TraversalStack {
        static constexpr int MaxSize = size;

        int len = 0;
        IdxT stack[size] = {};

        TraversalStack() = default;
        ~TraversalStack() = default;

        __host__ __device__ inline IdxT peek() const { return stack[len-1]; }
        __host__ __device__ void inline push(IdxT idx) { stack[len++] = idx; }
        __host__ __device__ inline void pop() { len--; }
    };
} // namespace stream::geo


#endif // __STREAM_TRAVERSAL_STACK_HPP__
