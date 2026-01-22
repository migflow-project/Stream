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

from subprocess import run
import os

def run_uniform2D():
    npoints = [10000, 50000, 100000, 500000, 1000000, 10000000]
    alphas = [0.024, 0.0084, 0.006, 0.0018, 0.0012, 0.00042]

    for npoint, alpha in zip(npoints, alphas):
        argv = [
            "./testcases/IMR26/uniform2D",
            "-n", str(npoint),
            "-a", str(alpha)
        ]
        print("="*70)
        print("Running command: ", " ".join(argv))
        run(argv)


def run_uniform3D():
    npoints = [10000, 50000, 100000, 500000, 1000000]
    alphas = [0.06, 0.036, 0.024, 0.0144, 0.0108]

    for npoint, alpha in zip(npoints, alphas):
        argv = [
            "./testcases/IMR26/uniform3D",
            "-n", str(npoint),
            "-a", str(alpha)
        ]
        print("="*70)
        print("Running command: ", " ".join(argv))
        run(argv)


def run_simulation_meshes():
    argv = [ "python", "./testcases/IMR26/run_on_simulation_meshes.py"]
    print("="*70)
    print("Running command: ", " ".join(argv))
    run(argv)


if __name__ == "__main__":

    # PYTHONPATH used for running python processes
    pythonpath=f"@CMAKE_BINARY_DIR@:{os.getenv("PYTHONPATH")}"
    os.environ["PYTHONPATH"] = pythonpath

    run_uniform2D()
    run_uniform3D()
    run_simulation_meshes()
