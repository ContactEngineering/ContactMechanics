#
# Copyright 2018-2019 Antoine Sanner
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
try:
    import numpy as np
    import time
    import math
    from PyCo.ContactMechanics import HardWall
    from PyCo.SolidMechanics import PeriodicFFTElasticHalfSpace
    from PyCo.SolidMechanics import FreeFFTElasticHalfSpace
    from PyCo.Topography import make_sphere,Topography
    from PyCo.System import make_system
    #from PyCo.Tools.Logger import screen
    from PyCo.ReferenceSolutions.Hertz import (radius_and_pressure,
                                               surface_displacements,
                                               surface_stress)
    from NuMPI.Optimization import LBFGS
    from NuMPI.Tools.Reduction import Reduction
    from mpi4py import MPI
    from PyCo.ContactMechanics import VDW82smoothMin, VDW82
    from PyCo.System import SmoothContactSystem
    from PyCo.Tools.NetCDF import NetCDFContainer

except ImportError as err:
    import sys
    print(err)
    sys.exit(-1)

_toplot=False
def test_wavy(comm, fftengine_type):

    n=32
    surf_res = (n,n)
    surf_size = (n,n)

    z0 = 1
    Es = 1

    R = 100
    w = 0.01*z0 * Es

    fftengine = fftengine_type((n, n), comm=comm)

    pnp = Reduction(comm=comm)

    inter = VDW82smoothMin(w * z0 ** 8 / 3, 16 * np.pi * w * z0 ** 2, gamma=w,pnp = pnp)

    # Parallel Topography Patch

    substrate = PeriodicFFTElasticHalfSpace(surf_res, young=Es,
                                            physical_sizes=surf_size, pnp=pnp)

    surface = Topography(
        np.cos(np.arange(0, n) * np.pi * 2. / n) * np.ones((n, 1)),
        size=surf_size)

    psurface = Topography(surface.heights(), size=surface.physical_sizes,
                          subdomain_location=substrate.topography_subdomain_location,
                          subdomain_resolution=substrate.subdomain_resolution,
                          periodic=True, pnp=pnp)

    system = SmoothContactSystem(substrate, inter, psurface)

    offsets = np.linspace(-2, 1, 50)

    force = np.zeros_like(offsets)

    nsteps = len(offsets)

    for i in range(nsteps):
        result = system.minimize_proxy(offsets[i], disp0=None,method=LBFGS,
                                       options=dict(gtol=1e-5, maxiter=100, maxls=10, pnp=pnp))
        assert result.success
        force[i] = system.compute_normal_force()
        #print("step {}".format(i))

    toPlot = comm.Get_rank() == 0 and _toplot

    if toPlot:
        import matplotlib
        #matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax  = plt.subplots()
        ax.set_xlabel("displacement")
        ax.set_ylabel("force")

        ax.plot(offsets, force)
        #plt.show(block=True)
        figname="MPI_Smoothcontact_tests.png"
        fig.savefig(figname)

        import subprocess
        subprocess.check_call("open {}".format(figname), shell=True)

        plt.show(block=True)

if __name__ == "__main__":
    from mpi4py import MPI
    _toplot = True
    test_wavy(MPI.COMM_WORLD)