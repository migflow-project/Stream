# 
# Stream - Copyright (C) <2025-2026>
# <Universite catholique de Louvain (UCL), Belgique>
# 
# List of the contributors to the development of Stream: see AUTHORS file.
# Description and complete License: see LICENSE file.
# 
# This file is part of Stream. Stream is free software:
# you can redistribute it and/or modify it under the terms of the GNU Lesser General
# Public License as published by the Free Software Foundation, either version 3
# of the License, or (at your option) any later version.
# 
# Stream is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Lesser General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public License along with Stream. 
# If not, see <https://www.gnu.org/licenses/>.
# 
import stream.numerics as stn
import numpy as np
import scipy.sparse as sp
from time import perf_counter

n = 50000   # Size of the matrix
target_sparsity = 0.001    # Percent of nonzero values in the matrix

nnz = int(n * (n * target_sparsity))
nnz_half = nnz // 2   # /2 because we generate the Lower triangulat
                      # matrix and then symmetrize

# To get the target sparsity, we generate a COO matrix with nnz/2 random triplets
np.random.seed(42)
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
t0 = perf_counter()
x = np.zeros(n, dtype=np.float32)
niter = cg.solve_jacobi(sys, prec, x)
print(f"Time for our solve : {perf_counter() - t0}")

# Solve with scipy
t0 = perf_counter()
prec = 1./Asp[np.diag_indices(n)]
prec = sp.csr_array((prec, (list(range(n)), list(range(n)))), shape=(n, n), dtype=np.float32)
x_sp, status = sp.linalg.cg(Asp, b, M = prec)
print(f"Time for scipy solve : {perf_counter() - t0}")

# Compare
allclose = np.allclose(x, x_sp)
mse = np.sqrt(np.mean((x - x_sp)**2))
abs_err = max(abs(x - x_sp))

print(f"Mean squared error between scipy and ours : {mse}. Maximum absolute error : {abs_err}. Allclose : {allclose}")
