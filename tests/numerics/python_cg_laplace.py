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
import matplotlib.pyplot as plt
from time import perf_counter

plot = False

# Solve a Laplace's equation with manufactured solution : u = sin(ax)cos(by)
#          div(grad(u)) = -(a^2 + b^2) sin(ax)cos(by)    on the interior of D 
#          u = sin(ax)cos(by)                            on the boundary of D
# With the domain D = [0, 1]^2
# 
# The discretization is a centered finite difference scheme on a grid. 
# The dof are stored at the nodes of the grid.

a = 50
b = 50
def u_analytical(xi, yj):
    ret = np.sin(a*xi) * np.cos(b*yj)
    return ret

n = 100   # number of grid points in x dimension
m = 100   # number of grid points in y dimension
ndof = n*m

x, dx = np.linspace(0, 1, n, endpoint=True, retstep=True)
y, dy = np.linspace(0, 1, m, endpoint=True, retstep=True)

row = []
col = []
val = []

bind = np.zeros(ndof)

# Boundary conditions : bottom/top
for i in range(n):
    ij = (i, 0)
    node = ij[0]*m + ij[1]
    row.append(node)
    col.append(node)
    val.append(1.0)
    bind[node] = u_analytical(x[ij[0]], y[ij[1]])

    ij = (i, m-1)
    node = ij[0]*m + ij[1]
    row.append(node)
    col.append(node)
    val.append(1.0)
    bind[node] = u_analytical(x[ij[0]], y[ij[1]])

# Boundary conditions : left/right
for j in range(1, m-1):
    ij = (0, j)
    node = ij[0]*m + ij[1]
    row.append(node)
    col.append(node)
    val.append(1.0)
    bind[node] = u_analytical(x[ij[0]], y[ij[1]])

    ij = (n-1, j)
    node = ij[0]*m + ij[1]
    row.append(node)
    col.append(node)
    val.append(1.0)
    bind[node] = u_analytical(x[ij[0]], y[ij[1]])

dx2_i = 1./(dx*dx)
dy2_i = 1./(dy*dy)
for i in range(1, n-1):
    for j in range(1, m-1):
        node = i*m + j

        if (i == 1):
            bind[node] += dx2_i * bind[node-m]
        else:
            row.append(node)
            col.append(node-m)
            val.append(-dx2_i)

        if (i == n-2):
            bind[node] += dx2_i * bind[node+m]
        else:
            row.append(node)
            col.append(node+m)
            val.append(-dx2_i)

        row.append(node)
        col.append(node)
        val.append(2.*dx2_i + 2.*dy2_i)

        if (j == 1):
            bind[node] += dy2_i * bind[node-1]
        else:
            row.append(node)
            col.append(node-1)
            val.append(-dy2_i)

        if (j == m-2):
            bind[node] += dy2_i * bind[node+1]
        else:
            row.append(node)
            col.append(node+1)
            val.append(-dy2_i)

        bind[node] += (a*a + b*b)*np.sin(a*x[i])*np.cos(b*y[j])

Acoo = sp.coo_array((val, (row, col)), shape=(ndof, ndof))
Asp = Acoo.tocsr()

# Generate the host CSR matrix
hcsr = stn.HostCSR(Asp.indptr, Asp.indices, Asp.data)
# Copy it into device
dcsr = stn.DeviceCSR(hcsr)

# Create the linear system with a random independant vector 
sys = stn.LinSys(dcsr, bind)

# Compute the preconditioner
prec = stn.PrecJacobi(dcsr)

# Initialize the Conjugate Gradient solver
cg = stn.SolverCG()

# Solve with our library
t0 = perf_counter()
u_disc_prec = np.zeros(ndof, dtype=np.float32)
niter = cg.solve_jacobi(sys, prec, u_disc_prec)
print(f"Time for our preconditioned solve : {perf_counter() - t0}. niter = {niter}")

t0 = perf_counter()
u_disc_noprec = np.zeros(ndof, dtype=np.float32)
niter = cg.solve(sys, u_disc_noprec)
print(f"Time for our non-preconditioned solve : {perf_counter() - t0}. niter = {niter}")

# Solve with scipy 
t0 = perf_counter()
u_sp, status = sp.linalg.cg(Asp, bind)
print(f"Time for our non-preconditioned scipy solve : {perf_counter() - t0}.")

u_real = np.zeros(ndof)
for i in range(n):
    for j in range(m):
        u_real[i*n+j] = u_analytical(x[i], y[j])

# Compare
mse_prec = np.sqrt(np.mean((u_disc_prec - u_real)**2))
abs_err_prec = max(abs(u_disc_prec - u_real))

mse_noprec = np.sqrt(np.mean((u_disc_noprec - u_real)**2))
abs_err_noprec = max(abs(u_disc_noprec - u_real))

mse_sp = np.sqrt(np.mean((u_sp - u_real)**2))
abs_err_sp = max(abs(u_sp - u_real))

print(f"[prec] Mean squared error between analytical and ours : {mse_prec}. Maximum absolute error : {abs_err_prec}.")
print(f"[noprec] Mean squared error between analytical and ours : {mse_noprec}. Maximum absolute error : {abs_err_noprec}.")
print(f"[scipy] Mean squared error between analytical and scipy : {mse_sp}. Maximum absolute error : {abs_err_sp}.")

if plot:
    Z = u_disc_prec.reshape((n, m))
    X, Y = np.meshgrid(x, y)
    plt.contourf(X, Y, Z)
    plt.show()
