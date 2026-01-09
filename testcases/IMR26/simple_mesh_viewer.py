import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.collections as mc
import matplotlib.patches as mp
import sys

fnode = sys.argv[1]
felem = sys.argv[2]

nodes = np.loadtxt(fnode, np.float32, delimiter=" ")
elems = np.loadtxt(felem, np.uint32, delimiter=" ")

fig, ax = plt.subplots()
ax.triplot(nodes[:, 0], nodes[:, 1], elems, linewidth=2)
ax.scatter(nodes[:, 0], nodes[:, 1])
ax.set_aspect("equal")
plt.show()
