# IMR26 benchmarks

For reproducibility, this directory contains the benchmarks and testcases shown 
in our IMR26 paper.

> [!NOTE] Note
> When rewriting the code for publication, I discovered some small performance losses due to the implementation. This cleaned up implementation should yield slightly better performances than explained in the paper. In addition, the performances strongly depends on the specifications of your hardware.
> To get a rough idea of the speedup you should obtain on your hardware with regards to mine, you can look at the number of cores on your card and divide it by 3072 (the number of CUDA cores on my RTX4060). For example the RTX6000 used in the paper has roughly 18k cores and (for large point-cloud) you can observe a factor of 5-6 between the timings on both cards. 

---

## Run all testcases at once 

The Python script `build/testcases/IMR26/run_all.py`, allows to run all the test-cases described in this README at once.
It will output each of the results in stdout. 

```bash 
cd build/
python ./testcases/IMR26/run_all.py 
```

## `uniform2D.cpp`

This testcase was used to get the "Ours" column in table 1.a of our paper

To see what command line options are available, run:
```bash 
$ cd build/                  # make sure to be in your build directory
$ ./testcases/IMR26/uniform2D --help
```

Values used in the paper :

| -n       |   -a    |
|----------|---------|
| 10000    | 0.024   |
| 50000    | 0.0084  |
| 100000   | 0.006   |
| 500000   | 0.0018  |
| 1000000  | 0.0012  |
| 10000000 | 0.00042 |

Example for running our 2D alpha-shape algorithm on 1M uniformly sampled points 
using an alpha value of 0.0012 and 10 repetitions.

```bash 
$ cd build/               # make sure to be in your build directory
$  ./testcases/IMR26/uniform2D -n 1000000 -a 0.0012 -r 5 | column -t
run  alpha     npoints  ntri     tlbvh      tashape    tcompress  ttot
0    0.001200  1000000  1877465  14.827297  18.336657  0.165415   33.163954
1    0.001200  1000000  1877465  5.223866   17.122290  0.003134   22.346156
2    0.001200  1000000  1877465  5.207062   17.139735  0.002787   22.346797
3    0.001200  1000000  1877465  5.209248   17.152754  0.002740   22.362002
4    0.001200  1000000  1877465  5.218465   17.134976  0.002700   22.353441
```

Description of the columns:

- `run`: the ID of the sample
- `alpha`: alpha value used for the alpha-shape
- `ntri`: number of simplices in the resulting mesh (triangles in 2D and tetrahedrons in 3D)
- `tlbvh`: time to construct the LBVH and query the *number* of neighbors, in milliseconds
- `tashape`: time to query the *ID* of the neighbors and to construct the alpha-shape mesh, in milliseconds
- `tcompress`: time to compress the arrays into CSR format for easier CPU-GPU communications, in milliseconds
- `ttot`: the total time, in milliseconds


## `uniform3D.cpp` 

This testcase was used to get the "Ours" column in table 1.b of our paper

To see what command line options are available, run:
```bash 
$ cd build/         # make sure to be in your build directory
$ ./testcases/IMR26/uniform3D --help
```

Values used in the paper :

| -n      |   -a    |
|---------|---------|
| 10000   | 0.06    |
| 50000   | 0.036   |
| 100000  | 0.024   |
| 500000  | 0.0144  |
| 1000000 | 0.0108  |

Example for running our 3D alpha-shape algorithm on 100k uniformly sampled points 
using an alpha value of 0.022 and 10 repetitions.

```bash 
$ cd build/         # make sure to be in your build directory
$  ./testcases/IMR26/uniform3D -n 100000 -a 0.022 -r 5 | column -t
run  alpha     npoints  ntri    tlbvh     tashape    tcompress  ttot
0    0.022000  100000   526040  7.069145  27.886839  0.187139   34.955984
1    0.022000  100000   526040  2.174104  25.823812  0.003399   27.997916
2    0.022000  100000   526040  2.161516  25.930698  0.003542   28.092214
3    0.022000  100000   526040  2.161072  26.005754  0.003862   28.166826
4    0.022000  100000   526040  2.157196  25.871720  0.003850   28.028916
```

Description of the columns:

- `run`: the ID of the sample
- `alpha`: alpha value used for the alpha-shape
- `ntri`: number of simplices in the resulting mesh (triangles in 2D and tetrahedrons in 3D)
- `tlbvh`: time to construct the LBVH and query the *number* of neighbors, in milliseconds
- `tashape`: time to query the *ID* of the neighbors and to construct the alpha-shape mesh, in milliseconds
- `tcompress`: time to compress the arrays into CSR format for easier CPU-GPU communications, in milliseconds
- `ttot`: the total time, in milliseconds


## `run_on_simulation_meshes.py` 

This script can be used to reproduce the validation testcases of section 7.1 to 7.3.
It is a Python script that will read the meshes from `build/input_meshes/`. And 
execute both the 2D-3D testcases using our library's Python API.

Note the `PYTHONPATH=$PYTHONPATH:.` at the start of the command to append the build directory to the `PYTHONPATH`
for this command. This will allow to use the Python's binding of our library.

Running the testcase :
```console 
$  cd build/
$  PYTHONPATH=.:$PYTHONPATH python testcases/IMR26/run_on_simulation_meshes.py 
[2D] Time to mesh './input_meshes/mesh_waterfall/mesh_waterfall_1e-03.bin' (4552 nodes, 8313 triangles) : 0.0008s
[...]
[2D] Time to mesh './input_meshes/mesh_dambreak/mesh_dambreak_1e-03.bin' (4201 nodes, 6771 triangles) : 0.0004s
[...]
[3D] Time to mesh './input_meshes/mesh_wine_glass.bin' (20037 nodes, 79049 tetrahedrons) : 0.0044s
```

This python script is supposed to unzip the dataset automatically, but
if you encounter errors of the form `Could not find input file: ...`, make sure
the archive `testcases/IMR26/IMR26_GPU_AlphaShape_dataset.zip` is unzipped 
in your build directory. I.e. there should be a directory `build/input_meshes/`.

The input point clouds are provided in binary to reduce the repository's size, but they can 
easily be read with the following Python command if you want to test other point-sets: 

```python 
# use a shape of (-1, 3) for 2D point clouds and (-1, 4) for 3D point clouds.
# The file is a simple binary array of x, y, (z), alpha components.
point_cloud = numpy.fromfile(filename, dtype=np.float32).reshape((-1, 3))   
```
