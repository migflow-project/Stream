# Stream : a GPU implementation of the PFEM for incompressible free-surface flows.

> [!NOTE] Stream 
> A continuous flow of fluid, data or instructions.

This library aims at providing a parallel implementation for all building blocks 
of the Particle Finite Element Method pipeline :

- Mesh computation
- Free-surface detection
- Mesh adaptation
- Assembly of the linear system of equations 
- Solution of the system

## Requirements 

The bare minimum to be able to compile and run the code is the following :

- CMake ($\geq$ 3.26)
- A C++ compiler supporting C++20 standard

Optional, but strongly recommended requirements include :

- a GPU compiler (CUDA/HIP, depending on your target hardware)

## Compilation 

```bash 
# Clone the repo and its dependencies
git clone git@git.immc.ucl.ac.be:tihonn/stream.git --recurse-submodules

# Create and enter build directory 
mkdir build && cd build

# Configure the project 
cmake .. [additional config options]
# e.g. : cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_TESTS=ON -DENABLE_CUDA=ON

# Compile the code. NOTE : -j is strongly recommended when compiling for GPU
# as it is much slower than classical CPU compilation
make -j
```

### Configuration options :

The configuration options are the following :

- `-DENABLE_TESTS` (default = OFF) : compile the tests 
- `-DENABLE_CUDA` (default = ON) : Compile for CUDA architecture. *Requires a CUDA compiler*.
- `-DENABLE_CUDA_ARCH` (default = native) : Compile for a given CUDA architecture.

The usual CMake configuration options such as `-DCMAKE_BUILD_TYPE` are also valid.

## Running Test-cases

## Project Structure

```text 
|- deps/                   # Contains dependencies               
   |- ava/                 # Required dependency : CPU/GPU portable compilation
|- include/                # Public include files
   |- geometry/            # Geometry module: spatial search structures, geometric primitives...
   |- mesh/                # Mesh module: mesh generation, mesh adaptation, size-fields...
   |- numerics/            # Numerics module: linear system, solvers (physics AGNOSTIC)
   |- fem/                 # fem module: matrix assembly (physics BASED)
|- src/                    # Source code
   |- core/
   |- geometry/
   |- mesh/ 
   |- numerics/
   |- fem/
|- tests/                  # Tests functionalities against known results, or for performance
   |- geometry/
   |- mesh/ 
   |- numerics/
|- testcases/              # Physical testcases (e.g. Hysing, Dam break...)
|- python/                 # Python API 
```
