import stream.numerics as stn
import numpy as np
import scipy.sparse as sp

n = 50000   # Size of the matrix
target_sparsity = 0.001    # Percent of nonzero values in the matrix

nnz = int(n * (n * target_sparsity))
nnz_half = nnz // 2   # /2 because we generate the Lower triangulat
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
t0 = perf_counter()
x = np.zeros(n, dtype=np.float32)
niter = cg.solve_jacobi(sys, prec, x)
print(f"Time for our solve : {perf_counter() - t0}")

# Solve with scipy
t0 = perf_counter()
x_sp, status = sp.linalg.cg(Asp, b)
print(f"Time for scipy solve : {perf_counter() - t0}")

# Compare
allclose = np.allclose(x, x_sp)
error = np.sqrt(np.sum((x - x_sp)**2))

print(f"Mean squared error between scipy and ours : {error}. Allclose : {allclose}")
