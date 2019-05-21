#
# Copyright 2019 Antoine Sanner
# 
# ### MIT license
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#


import time

starttime=time.time()
import numpy as np
from PyCo.SolidMechanics import FreeFFTElasticHalfSpace
from PyCo.Topography import make_sphere

from FFTEngine import PFFTEngine
from NuMPI.Optimization import LBFGS
from NuMPI.Tools.Reduction import Reduction
from PyCo.ContactMechanics import VDW82smoothMin
from PyCo.System import SmoothContactSystem


from NuMPI.IO import save_npy
from NuMPI import MPI

import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

try:
    n = int(sys.argv[1])
except Exception:
    n = 128

import matplotlib.pyplot as plt

fig, (axt, axit) = plt.subplots(2, 1, sharex=True)

for n in [128,256,512]:
    # sphere radius:
    r_s = 10.0
    # contact radius
    r_c = .2
    # peak pressure
    p_0 = 2.5
    # equivalent Young's modulus
    E_s = 102.#102.
    # work of adhesion
    w = 1.0
    # tolerance for optimizer
    tol = 1e-12
    # tolerance for contact area
    gap_tol = 1e-6

    nx, ny = n, n
    sx = 21.0

    z0 = 0.05 # needed to get small tolerance, but very very slow

    fftengine = PFFTEngine((2*nx, 2*ny), comm=comm)
    pnp = Reduction(comm=comm)

    # the "Min" part of the potential (linear for small z) is needed for the LBFGS without bounds
    inter = VDW82smoothMin(w * z0 ** 8 / 3, 16 * np.pi * w * z0 ** 2, gamma=w, pnp = pnp)

    # Parallel Topography Patch

    substrate = FreeFFTElasticHalfSpace((nx,ny), young=E_s, size=(sx,sx), fftengine=fftengine, pnp=pnp)
    #print(substrate._comp_resolution)
    #print(fftengine.domain_resolution)


    surface = make_sphere(radius=r_s, resolution=(nx, ny), size=(sx, sx),
                          subdomain_location=substrate.topography_subdomain_location,
                          subdomain_resolution=substrate.topography_subdomain_resolution,
                          pnp=pnp,
                          standoff=float('inf'))
    ext_surface = make_sphere(r_s, (2 * nx, 2 * ny), (2 * sx, 2 * sx),
                              centre=(sx / 2, sx / 2),
                              subdomain_location=substrate.subdomain_location,
                              subdomain_resolution=substrate.subdomain_resolution,
                              pnp=pnp,
                              standoff=float('inf'))
    system = SmoothContactSystem(substrate, inter, surface)

    penetration = 0

    disp0 = ext_surface.heights() + penetration
    disp0 = np.where(disp0 > 0, disp0, 0)
    disp0 = system.shape_minimisation_input(disp0)

    maxcors = [5, 10, 20]
    times = [None] * len(maxcors)
    nits = [None] * len(maxcors)

    for i, maxcor in enumerate(maxcors):
        starttime =time.time()
        result = LBFGS(system.objective(penetration, gradient=True), disp0, jac=True, pnp=pnp, maxcor=maxcor, gtol=1e-6 * abs(w/z0))
        times[i] = time.time() - starttime
        nits[i]=result.nit
        #result = system.minimize_proxy(offsets[i], disp0=None,method = LBFGS,options=dict(gtol = 1e-3, maxiter =100,maxls=10))
        print(result.nit)
        print(times[i])
        converged = result.success
        assert converged


    if rank == 0:

        axt.plot(maxcors, times, label="n={}".format(n))
        axit.plot(maxcors, nits, label="n={}".format(n))


axit.set_xlabel("# gradients stored (-)")
axt.set_ylabel("execution time (s)")
axit.set_ylabel("# of iterations")
axt.legend()
fig.savefig("influence_maxcor.png")

