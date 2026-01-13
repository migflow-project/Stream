import numpy as np 
import matplotlib.pyplot as plt 
import stream.mesh as stm
import pathlib
from time import perf_counter

waterfall_files = [
    "./input_meshes/mesh_waterfall/mesh_waterfall_1e-03.bin",
    "./input_meshes/mesh_waterfall/mesh_waterfall_1e-04.bin",
    "./input_meshes/mesh_waterfall/mesh_waterfall_7e-05.bin",
    "./input_meshes/mesh_waterfall/mesh_waterfall_5e-05.bin",
    "./input_meshes/mesh_waterfall/mesh_waterfall_4e-05.bin",
    "./input_meshes/mesh_waterfall/mesh_waterfall_3e-05.bin",
    "./input_meshes/mesh_waterfall/mesh_waterfall_2e-05.bin",
    "./input_meshes/mesh_waterfall/mesh_waterfall_1e-05.bin"
]

dambreak_files = [
    "./input_meshes/mesh_dambreak/mesh_dambreak_0.0005.bin",
    "./input_meshes/mesh_dambreak/mesh_dambreak_4e-05.bin",
    "./input_meshes/mesh_dambreak/mesh_dambreak_5e-05.bin",
    "./input_meshes/mesh_dambreak/mesh_dambreak_0.0003.bin",
    "./input_meshes/mesh_dambreak/mesh_dambreak_0.0001.bin",
    "./input_meshes/mesh_dambreak/mesh_dambreak_7e-05.bin",
    "./input_meshes/mesh_dambreak/mesh_dambreak_3e-05.bin",
    "./input_meshes/mesh_dambreak/mesh_dambreak_0.001.bin"
]

files_2D = []
files_2D.extend(waterfall_files)
files_2D.extend(dambreak_files)

wine_glass_files = [
    "./input_meshes/mesh_wine_glass.bin"
]
files_3D = []
files_3D.extend(wine_glass_files)

# Use the same AlphaShape2D object
ashape = stm.AlphaShape2D()
for file in files_2D:

    if (not pathlib.Path(file).exists()):
        print(f"Could not find input file '{file}'")
        continue

    # Read binary files
    point_cloud = np.fromfile(file, dtype=np.float32).reshape((-1, 3))
    coords = point_cloud[:, :2]
    alpha = point_cloud[:, 2]


    nruns = 10
    start = perf_counter()
    for run in range(nruns):
        # Compute total compute time : computing lbvh + neighbors + triangulation + filter + compression
        ashape.set_nodes(coords, alpha) # lbvh + 1st pass neighbors
        ashape.compute()                # 2nd pass neighbors + triangulation + filter + data compression
    end = perf_counter()

    time_per_run = (end - start) / nruns

    # Retrieve output
    # WARNING : the triangles use indices of the morton-ordered objects
    #           Hence using the "coords" array will yield nonsensical graphs
    #           To have sensible visualisations, retrieve the morton-ordered objects 
    #           using ashape.get_ordered_nodes()
    tri = ashape.get_elem()
    nodes = ashape.get_ordered_nodes()

    print(f"[2D] Time to mesh '{file}' ({len(nodes)} nodes, {len(tri)} triangles) : {time_per_run:.4f}s")

    # Plot if the mesh size is not too large
    # Matplotlib freezes on large input set
    if len(nodes) < 50000:
        fig, ax = plt.subplots()

        ax.set_aspect("equal")
        ax.triplot(nodes[:, 0], nodes[:, 1], tri, linewidth=1)

        plt.show()
        


# Use the same AlphaShape3D object
ashape = stm.AlphaShape3D()
for file in files_3D:

    if (not pathlib.Path(file).exists()):
        print(f"Could not find input file '{file}'")
        continue

    # Read binary files
    point_cloud = np.fromfile(file, dtype=np.float32).reshape((-1, 4))
    coords = point_cloud[:, :3]
    alpha = point_cloud[:, 3]


    nruns = 10
    start = perf_counter()
    for run in range(nruns):
        # Compute total compute time : computing lbvh + neighbors + triangulation + filter + compression
        ashape.set_nodes(coords, alpha) # lbvh + 1st pass neighbors
        ashape.compute()                # 2nd pass neighbors + triangulation + filter + data compression
    end = perf_counter()

    time_per_run = (end - start) / nruns

    # Retrieve output
    # WARNING : the triangles use indices of the morton-ordered objects
    #           Hence using the "coords" array will yield nonsensical graphs
    #           To have sensible visualisations, retrieve the morton-ordered objects 
    #           using ashape.get_ordered_nodes()
    tri = ashape.get_elem()
    nodes = ashape.get_ordered_nodes()

    print(f"[3D] Time to mesh '{file}' ({len(nodes)} nodes, {len(tri)} tetrahedrons) : {time_per_run:.4f}s")
