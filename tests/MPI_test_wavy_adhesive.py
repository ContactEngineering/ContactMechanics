try:
    import unittest
    import numpy as np
    import time
    import math
    from PyCo.ContactMechanics import HardWall
    from PyCo.SolidMechanics import PeriodicFFTElasticHalfSpace
    from PyCo.SolidMechanics import FreeFFTElasticHalfSpace
    from PyCo.Topography import Sphere,UniformNumpyTopography
    from PyCo.System import SystemFactory
    #from PyCo.Tools.Logger import screen
    from PyCo.ReferenceSolutions.Hertz import (radius_and_pressure,
                                               surface_displacements,
                                               surface_stress)
    from FFTEngine import PFFTEngine
    from PyLBFGS.MPI_LBFGS_Matrix_H import LBFGS
    from PyLBFGS.Tools.ParallelNumpy import ParallelNumpy
    from mpi4py import MPI
    from PyCo.ContactMechanics import VDW82smoothMin, VDW82
    from PyCo.System import SmoothContactSystem
    from PyCo.Tools.NetCDF import NetCDFContainer

except ImportError as err:
    import sys
    print(err)
    sys.exit(-1)


def test_wavy():
    comm = MPI.COMM_WORLD

    n=32
    surf_res = (n,n)
    surf_size = (n,n)

    z0 = 1
    Es = 1

    R = 100
    w = 0.0001*z0 * Es

    fftengine = PFFTEngine((n,n),comm=comm)

    inter = VDW82smoothMin(w * z0 ** 8 / 3, 16 * np.pi * w * z0 ** 2, gamma=w) # TODO: Maybe interaction will also need to be parallel !

    # Parallel Topography Patch

    substrate = PeriodicFFTElasticHalfSpace(surf_res,young=Es,size=surf_size,fftengine=fftengine)

    class Parallel_Topography(): # Just some Temp implementation of the interface
        def __init__(self,surface,fftengine):
            self.surface = surface
            self.subdomain_resolution = fftengine.subdomain_resolution # TODO: FreeElastHS: sometimes the subdomain is emptym, comp_slice ?
            self.subdomain_slice = fftengine.subdomain_slice

            self.domain_resolution = fftengine.domain_resolution
            self.resolution = self.surface.resolution

        def array(self,*args,**kwargs):
            return self.surface.array()[self.subdomain_slice]


    surface =UniformNumpyTopography(np.cos(np.arange(0,n) * np.pi * 2. /n ) * np.ones((n,1)),size = surf_size)
    psurface = Parallel_Topography(surface, fftengine)

    system = SmoothContactSystem(substrate, inter, psurface)

    offsets = np.linspace(-2,1,50)

    force = np.zeros_like(offsets)

    nsteps = len(offsets)


    for i in range(nsteps):


        result = system.minimize_proxy(offsets[i], disp0=None,method = LBFGS,options=dict(gtol = 1e-3, maxiter =100,maxls=10))


        assert result.success

        force[i] = system.compute_normal_force()
        print("step {}".format(i))

    toPlot = comm.Get_rank() == 0 and True

    if toPlot:
        import matplotlib
        #matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax  = plt.subplots()
        ax.set_xlabel("displacement")
        ax.set_ylabel("force")

        ax.plot(offsets, force)
        #plt.show(block=True)
        fig.savefig("MPI_Smoothcontact_tests.png")

if __name__ == "__main__":
    test_wavy()