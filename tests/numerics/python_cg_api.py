import stream.numerics as stn
import numpy as np
import scipy.sparse as sp
from time import perf_counter

n = 1000   # Size of the matrix
target_sparsity = 0.01    # Percent of nonzero values in the matrix

# To get the sparsity we want, generate a matrix with entries 
# taken from a uniform [0, 1] distribution and discard any 
# entries greater than the target sparsity
A = np.random.uniform(0, 1, size=(n, n)).astype(np.float32)
A = A@A.T

# This is to reduce the conditioning number
A[np.diag_indices(n)] += n/100

Asp = sp.csr_array(A, dtype=np.float32)

hcsr = stn.HostCSR(Asp.indptr, Asp.indices, Asp.data)
dcsr = stn.DeviceCSR(hcsr)
prec = stn.PrecJacobi(dcsr)
b = np.random.uniform(0, 1, size=(n,)).astype(np.float32)
sys = stn.LinSys(dcsr, b)

cg = stn.SolverCG()

t0 = perf_counter()
x = np.zeros(n, dtype=np.float32)
niter = cg.solve_jacobi(sys, prec, x)
print(f"Time for our solve : {perf_counter() - t0}")

t0 = perf_counter()
x_sp = sp.linalg.spsolve(Asp, b)
print(f"Time for scipy solve : {perf_counter() - t0}")

allclose = np.allclose(x, x_sp)
error = np.sqrt(np.sum((x - x_sp)**2))

print(f"Conditioning : {np.linalg.cond(A)}. Mean squared error between scipy and ours : {error}. Allclose : {allclose}")
