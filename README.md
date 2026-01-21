# Stream : a GPU implementation of the PFEM for incompressible free-surface flows.

In order to reproduce benchmarks shown in our IMR26 paper, please refer to 
[`./testcases/IMR26/README.md`](./testcases/IMR26/README.md) **after successful compilation of the project.**

> [!NOTE] Stream 
> A continuous flow of fluid, data or instructions.

This library aims at providing a parallel implementation for all building blocks 
of the Particle Finite Element Method pipeline:

- Mesh computation and adaptation
- Free-surface detection
- Assembly of the linear system of equations 
- Solution of the system

Currently, major updates to this repository occur when we publish our results 
in papers. However, please feel free to open an issue if you encounter
bugs or would like to request a feature for future updates.

## Requirements 

The bare minimum to be able to compile and run the code is the following :

- CMake 
- A C++ compiler supporting C++20 standard
- git-lfs, to be able to download the datasets.

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
git clone git@github.com:migflow-project/Stream.git --recurse-submodules

# If you downloaded from a tag/release, make sure the submodules are up-to-date 
git submodule update --recursive --init

# Make sure you have installed the "git-lfs" extension
# e.g. To install it on Arch : sudo pacman -Sy git-lfs 
git lfs install        # Setup git-lfs for the project
git lfs pull           # Download the datasets

cd Stream

# Create and enter build directory 
mkdir build && cd build

# Configure the project 
cmake .. [additional config options]
# Recommended for CPU: cmake .. -DCMAKE_BUILD_TYPE=Release
# Recommended for CUDA: cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON
# Recommended for HIP: cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_HIP=ON

# Compile the code. NOTE : -j is strongly recommended when compiling for GPU
# as it is much slower than classical CPU compilation
make -j
``` 

If you want to install the library system-wide, or in the prefix you gave CMake, run:
```bash
# (optional) System-wide install of the library 
sudo make install
```

### Configuration options :

The configuration options are the following :

- `-DENABLE_TESTS` (default = OFF) : compile the tests 
- `-DENABLE_CUDA` (default = OFF) : Compile for CUDA architecture. *Requires a CUDA compiler*.
- `-DENABLE_HIP` (default = OFF) : Compile for HIP architecture. *Requires a HIP compiler*.
- `-DENABLE_ARCH` (default = native) : Compile for a given architecture (valid for any target compilation).
- `-DAVA_ENABLE_TBB` (default = on if TBB is found, off otherwise) : Use TBB as a CPU parallelization.

The usual CMake configuration options such as `-DCMAKE_BUILD_TYPE` are also valid.

## Python bindings 

Once you've installed the library on your system, you can also use the Python bindings 
in the library `stream`. 

If Python does not find the `stream` package, make sure the directory in which
`stream` is located is in the environment variable `PYTHONPATH`. E.g. if `stream` 
is located in `path/to/directory/stream` make sure `path/to/directory/` is in the 
`PYTHONPATH` by running the command:
```bash
export PYTHONPATH=$PYTHONPATH:path/to/directory
```

Note that this will only modify the `PYTHONPATH` for the current terminal session. 
To avoid executing this command everytime you start a new terminal, you can put this command in your shell configuration file (`.bashrc`, `.zshrc`, ...)

## Running Test-cases

In order to reproduce benchmarks shown in our IMR26 paper, refer to the instructions in [`./testcases/IMR26/README.md`](./testcases/IMR26/README.md).

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
