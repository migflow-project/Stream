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
