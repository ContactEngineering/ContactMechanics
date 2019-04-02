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
    import unittest
    import numpy as np
    import time
    import math
    from PyCo.ContactMechanics import HardWall
    from PyCo.SolidMechanics import PeriodicFFTElasticHalfSpace
    from PyCo.SolidMechanics import FreeFFTElasticHalfSpace
    from PyCo.Topography import make_sphere
    from PyCo.System import make_system
    #from PyCo.Tools.Logger import screen
    from PyCo.ReferenceSolutions.Hertz import (radius_and_pressure,
                                               surface_displacements,
                                               surface_stress)
    from FFTEngine import PFFTEngine
    from NuMPI.Optimization import LBFGS
    from NuMPI.Tools.ParallelNumpy import ParallelNumpy  # TODO: This should be explicitly from NuMPI
    from mpi4py import MPI
    from PyCo.ContactMechanics import VDW82smoothMin, VDW82
    from PyCo.System import SmoothContactSystem
    from PyCo.Tools.NetCDF import NetCDFContainer
    from PyCo.Topography import Topography

except ImportError as err:
    import sys
    print(err)
    sys.exit(-1)


def test_smoothsphere():
    comm = MPI.COMM_WORLD

    n=4
    surf_res = (n,n)
    surf_size = (n,n)

    z0 = 1
    Es = 1

    R = 100
    w = 0.0001*z0 * Es

    fftengine = PFFTEngine((2*n, 2*n), comm=comm)
    pnp =ParallelNumpy(comm=comm)
    inter = VDW82smoothMin(w * z0 ** 8 / 3, 16 * np.pi * w * z0 ** 2, gamma=w, pnp = pnp)

    # Parallel Topography Patch

    substrate = FreeFFTElasticHalfSpace(surf_res, young=Es, size=surf_size, fftengine=fftengine, pnp=pnp)
    print(substrate._comp_resolution)
    print(fftengine.domain_resolution)

    # TODO; now it should be alright
    #class Parallel_Topography(): # Just some Temp implementation of the interface
    #    def __init__(self,surface,fftengine):
    #        self.surface = surface
    #        self.subdomain_resolution = fftengine.subdomain_resolution # TODO: FreeElastHS: sometimes the subdomain is empty, comp_slice ?
    #        self.subdomain_slice = fftengine.subdomain_slice#

        #     self.domain_resolution = fftengine.domain_resolution
        #     self.resolution = self.surface.resolution
        #
        # def array(self,*args,**kwargs):
        #     return self.surface.heights()[self.subdomain_slice]


    surface = make_sphere(radius = R, resolution = surf_res ,size =surf_size)
    psurface = Topography(surface.heights(),
                          subdomain_location=substrate.topography_subdomain_location,
                          subdomain_resolution=substrate.topography_subdomain_resolution,
                            size=surface.size,pnp = pnp)

    system = SmoothContactSystem(substrate, inter, psurface)

    offsets = np.linspace(-2,1,50)

    force = np.zeros_like(offsets)

    nsteps = len(offsets)

    for i in range(nsteps):

        disp0 = np.zeros(substrate.subdomain_resolution)
        disp0 = system.shape_minimisation_input(disp0)

        result = LBFGS(system.objective(offsets[i],gradient=True),disp0,jac = True,pnp = pnp)

        #result = system.minimize_proxy(offsets[i], disp0=None,method = LBFGS,options=dict(gtol = 1e-3, maxiter =100,maxls=10))


        assert result.success

        force[i] = system.compute_normal_force()
        print("step {}".format(i))

    toPlot = comm.Get_rank() == 0 and True

    if toPlot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax  = plt.subplots()
        ax.set_xlabel("displacement")
        ax.set_ylabel("force")

        ax.plot(offsets, force)
        #plt.show(block=True)
        fig.savefig("MPI_Smoothcontact_tests.png")

if __name__ == "__main__":
    test_smoothsphere()