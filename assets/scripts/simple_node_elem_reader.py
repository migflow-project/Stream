import numpy as np 
import matplotlib.pyplot as plt 
import sys

fnode = sys.argv[1]
felem = sys.argv[2]
fnode_complete = sys.argv[3]

nodes = np.loadtxt(fnode, np.float32, delimiter=" ")
elems = np.loadtxt(felem, np.uint32, delimiter=" ")

node_complete = np.loadtxt(fnode_complete, np.bool)
node_complete = np.hstack([node_complete, [1, 1, 1]])

plt.triplot(nodes[:, 0], nodes[:, 1], elems)
# plt.scatter(nodes[:, 0], nodes[:, 1], c=node_complete)
plt.show()

