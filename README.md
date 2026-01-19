# Stream : a GPU implementation of the PFEM for incompressible free-surface flows.

> [!NOTE] Stream 
> A continuous flow of fluid, data or instructions.

This library aims at providing a parallel implementation for all building blocks 
of the Particle Finite Element Method pipeline :

- Mesh computation and adaptation
- Free-surface detection
- Assembly of the linear system of equations 
- Solution of the system

In order to reproduce benchmarks shown in our IMR26 paper, please refer to 
[`./testcases/IMR26/README.md`](./testcases/IMR26/README.md) after successful compilation of the project.

## Requirements 

The bare minimum to be able to compile and run the code is the following :

- CMake 
- A C++ compiler supporting C++20 standard

Optional, but strongly recommended requirements include :

- a GPU compiler (CUDA/HIP, depending on your target hardware)
- cub if you use CUDA / hipcub is you use AMD / TBB for CPU multithreading

The library has been tested on Linux for CPU and CUDA architectures. 

## Compilation 

The following steps ensure that you download both the source code and its only 
required dependency (AVA), compile and install them on your system. 
The installation process also installs the Python API.

If you want to use CUDA/HIP, make sure to follow their installation instructions 
as well.

```bash 
# Clone the repo and its dependencies
git clone git@git.immc.ucl.ac.be:tihonn/stream.git --recurse-submodules

# Create and enter build directory 
mkdir build && cd build

# Configure the project 
cmake .. [additional config options]
# Recommended for sequential CPU : cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=OFF -DENABLE_TBB=OFF
# Recommended for CUDA : cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON

# Compile the code. NOTE : -j is strongly recommended when compiling for GPU
# as it is much slower than classical CPU compilation
make -j

# (optional) System-wide install of the library 
sudo make install
```

### Configuration options :

The configuration options are the following :

- `-DENABLE_TESTS` (default = OFF) : compile the tests 
- `-DENABLE_CUDA` (default = ON) : Compile for CUDA architecture. *Requires a CUDA compiler*.
- `-DENABLE_ARCH` (default = native) : Compile for a given architecture (valid for any target compilation).

The usual CMake configuration options such as `-DCMAKE_BUILD_TYPE` are also valid.

## Python bindings 

Once you've installed the library on your system, you can also use the Python bindings 
in the library `stream`. 

If Python does not find the `stream` package, make sure the directory in which
`stream` is located in the environment variable `PYTHONPATH`. E.g. if `stream` 
is located in `path/to/directory/stream` make sure `path/to/directory/` is in the 
`PYTHONPATH` by running the command :
```bash
export PYTHONPATH=$PYTHONPATH:path/to/directory
```

Example of simple system solve :
```python 
import stream.numerics as stn
import numpy as np
import scipy.sparse as sp

n = 50000   # Size of the matrix
target_sparsity = 0.001    # Percent of nonzero values in the matrix

nnz = int(n * (n * target_sparsity))
nnz_half = nnz // 2   # /2 because we generate the Lower triangular
                      # matrix and then symmetrize

# To get the target sparsity, we generate a COO matrix with nnz/2 random triplets
r = np.random.randint(0, n, size=(nnz_half,))
c = np.random.randint(0, n, size=(nnz_half,))
v = np.random.uniform(0, 1, size=(nnz_half,))

Acoo = sp.coo_array((v, (r, c)), shape=(n, n), dtype=np.float32)
Acoo = 0.5 * (Acoo + Acoo.T)                    # Symmetrize
Acoo[np.diag_indices(n)] += Acoo @ np.ones(n)   # Make diagonally dominant

# Transform to CSR for solve
Asp = sp.csr_array(Acoo, dtype=np.float32)

# Generate the host CSR matrix
hcsr = stn.HostCSR(Asp.indptr, Asp.indices, Asp.data)
# Copy it into device
dcsr = stn.DeviceCSR(hcsr)

# Create the linear system with a random independant vector 
b = np.random.uniform(0, 1, size=(n,)).astype(np.float32)
sys = stn.LinSys(dcsr, b)

# Compute the preconditioner
prec = stn.PrecJacobi(dcsr)

# Initialize the Conjugate Gradient solver
cg = stn.SolverCG()

# Solve with our library
x = np.zeros(n, dtype=np.float32)
niter = cg.solve_jacobi(sys, prec, x)
```

## Running Test-cases

## Project Structure

```text 
├── assets                # Data / scripts
│   ├── geofiles
│   └── scripts
├── AUTHORS.TXT
├── CMakeLists.txt
├── Config.cmake.in
├── COPYING.TXT
├── LICENSE.TXT
├── deps                  # Dependency directory
│   └── ava         
├── include               # Public include files
│   ├── CMakeLists.txt
│   ├── fem             
│   ├── geometry       
│   ├── mesh             
│   └── numerics
├── python                # Python bindings
│   ├── CMakeLists.txt
│   ├── __init__.py
│   ├── mesh
│   └── numerics
├── README.md
├── src                   # Source files for each module
│   ├── CMakeLists.txt
│   ├── core
│   ├── fem
│   ├── geometry
│   ├── mesh
│   ├── numerics
│   └── utils
├── testcases             # Physics testcases / comparison with other methods
│   ├── CMakeLists.txt
│   └── IMR26                 # The benchmarks/validation of our IMR26 paper
└── tests                 # Some quick tests to ensure parts of the library runs/produce consistant output
    ├── CMakeLists.txt
    ├── geometry
    ├── mesh
    └── numerics
```
