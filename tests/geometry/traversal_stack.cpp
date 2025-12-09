#include "traversal_stack.hpp"
#include <cstdio>

using Stack = stream::geo::TraversalMinHeap<int, float, 32>;

int main(void) {

    Stack stack;

    stack.push(10, 10.f);
    stack.push(9, 9.f);
    stack.push(8, 8.f);
    stack.push(7, 7.f);
    stack.push(6, 6.f);
    stack.push(5, 5.f);
    stack.push(4, 4.f);
    stack.push(3, 3.f);
    stack.push(2, 2.f);
    stack.push(1, 1.f);
    stack.push(0, 0.f);

    int len = stack.len;
    for (int i = 0; i < len; i++){
        Stack::Pair p = stack.peek();
        stack.pop();
        printf("Element %d = %d, %f\n", i, p.first, p.second);
    }

    return 0;
}
