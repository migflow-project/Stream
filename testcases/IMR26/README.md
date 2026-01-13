# IMR26 benchmarks

For reproducibility, this directory contains the benchmarks and testcases shown 
in our IMR26 paper.

> [!NOTE] When rewriting the code for publication, I discovered some small 
> performance losses due to the implementation. This cleaned up implementation 
> should hence perform slightly better than explained in the paper.

In order to be able to use the Python bindings, you can either :

- Add the `build/stream/` directory to your PYTHONPATH : `export PYTHONPATH=$PYTHONPATH:<path to build/stream`
- Install the library system-wide : `sudo make install`

## `uniform2D.cpp`

This testcase was used to get the "Ours" column in table 1.a of our paper

To see what command line options are available, run:
```console 
$ ./testcases/IMR26/uniform2D --help
```

Example for running our 2D alpha-shape algorithm on 1M uniformly sampled points 
using an alpha value of 0.0012 and 10 repetitions.

```console 
$  ./testcases/IMR26/uniform2D -n 1000000 -a 0.0012 -r 10 | column -t
init 2244.595422 ms
RNG  seed      :        42                                        
run  alpha     npoints  ntri     tlbvh      tashape    tcompress  ttot
0    0.001200  1000000  1877465  14.827297  18.336657  0.165415   33.163954
1    0.001200  1000000  1877465  5.223866   17.122290  0.003134   22.346156
2    0.001200  1000000  1877465  5.207062   17.139735  0.002787   22.346797
3    0.001200  1000000  1877465  5.209248   17.152754  0.002740   22.362002
4    0.001200  1000000  1877465  5.218465   17.134976  0.002700   22.353441
5    0.001200  1000000  1877465  5.210977   17.039915  0.003149   22.250892
6    0.001200  1000000  1877465  4.765141   15.766600  0.002671   20.531741
7    0.001200  1000000  1877465  4.686002   15.930445  0.002796   20.616447
8    0.001200  1000000  1877465  4.924608   16.495338  0.002781   21.419946
9    0.001200  1000000  1877465  5.026306   16.781896  0.002677   21.808202
```


## `uniform3D.cpp` 

This testcase was used to get the "Ours" column in table 1.b of our paper

To see what command line options are available, run:
```console 
$ ./testcases/IMR26/uniform3D --help
```

Example for running our 3D alpha-shape algorithm on 100k uniformly sampled points 
using an alpha value of 0.022 and 10 repetitions.

```console 
$  ./testcases/IMR26/uniform3D -n 100000 -a 0.022 -r 10 | column -t
init 202.356052 ms
RNG  seed      :        42                                      
run  alpha     npoints  ntri    tlbvh     tashape    tcompress  ttot
0    0.022000  100000   526040  7.069145  27.886839  0.187139   34.955984
1    0.022000  100000   526040  2.174104  25.823812  0.003399   27.997916
2    0.022000  100000   526040  2.161516  25.930698  0.003542   28.092214
3    0.022000  100000   526040  2.161072  26.005754  0.003862   28.166826
4    0.022000  100000   526040  2.157196  25.871720  0.003850   28.028916
5    0.022000  100000   526040  2.172017  26.129545  0.003880   28.301562
6    0.022000  100000   526040  2.172969  26.045723  0.004060   28.218692
7    0.022000  100000   526040  2.166163  25.886757  0.003319   28.052920
8    0.022000  100000   526040  2.161398  25.889633  0.004317   28.051031
9    0.022000  100000   526040  2.173429  25.784654  0.003371   27.958083
```

## `run_on_simulation_meshes.py` 

This script can be used to reproduce the validation testcases of section 7.1 to 7.3.

It is a Python script that will read the meshes from `./input_meshes/`. And 
execute both the 2D-3D testcases using our library's Python API.


```console 
$  python run_on_simulation_meshes.py 
[2D] Time to mesh './input_meshes/mesh_waterfall/mesh_waterfall_1e-03.bin' (4552 nodes, 8313 triangles) : 0.0008s
[2D] Time to mesh './input_meshes/mesh_waterfall/mesh_waterfall_1e-04.bin' (17889 nodes, 31566 triangles) : 0.0010s
[2D] Time to mesh './input_meshes/mesh_waterfall/mesh_waterfall_7e-05.bin' (26494 nodes, 47602 triangles) : 0.0012s
[2D] Time to mesh './input_meshes/mesh_waterfall/mesh_waterfall_5e-05.bin' (40268 nodes, 71910 triangles) : 0.0017s
[2D] Time to mesh './input_meshes/mesh_waterfall/mesh_waterfall_4e-05.bin' (50983 nodes, 91837 triangles) : 0.0019s
[2D] Time to mesh './input_meshes/mesh_waterfall/mesh_waterfall_3e-05.bin' (72432 nodes, 129419 triangles) : 0.0024s
[2D] Time to mesh './input_meshes/mesh_waterfall/mesh_waterfall_2e-05.bin' (109075 nodes, 197681 triangles) : 0.0037s
[2D] Time to mesh './input_meshes/mesh_waterfall/mesh_waterfall_1e-05.bin' (227101 nodes, 413066 triangles) : 0.0060s
[2D] Time to mesh './input_meshes/mesh_dambreak/mesh_dambreak_1e-03.bin' (4201 nodes, 6771 triangles) : 0.0004s
[2D] Time to mesh './input_meshes/mesh_dambreak/mesh_dambreak_5e-04.bin' (7448 nodes, 11759 triangles) : 0.0005s
[2D] Time to mesh './input_meshes/mesh_dambreak/mesh_dambreak_3e-04.bin' (13850 nodes, 22427 triangles) : 0.0006s
[2D] Time to mesh './input_meshes/mesh_dambreak/mesh_dambreak_1e-04.bin' (54513 nodes, 92913 triangles) : 0.0015s
[2D] Time to mesh './input_meshes/mesh_dambreak/mesh_dambreak_7e-05.bin' (83641 nodes, 144664 triangles) : 0.0022s
[2D] Time to mesh './input_meshes/mesh_dambreak/mesh_dambreak_5e-05.bin' (125825 nodes, 219361 triangles) : 0.0032s
[2D] Time to mesh './input_meshes/mesh_dambreak/mesh_dambreak_4e-05.bin' (163053 nodes, 286460 triangles) : 0.0040s
[2D] Time to mesh './input_meshes/mesh_dambreak/mesh_dambreak_3e-05.bin' (228432 nodes, 403525 triangles) : 0.0058s
[3D] Time to mesh './input_meshes/mesh_wine_glass.bin' (20037 nodes, 79049 tetrahedrons) : 0.0044s
```
