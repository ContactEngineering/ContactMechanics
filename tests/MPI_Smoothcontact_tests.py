try:
    import unittest
    import numpy as np
    import time
    import math
    from PyCo.ContactMechanics import HardWall
    from PyCo.SolidMechanics import PeriodicFFTElasticHalfSpace
    from PyCo.SolidMechanics import FreeFFTElasticHalfSpace
    from PyCo.Topography import Sphere
    from PyCo.System import SystemFactory
    #from PyCo.Tools.Logger import screen
    from PyCo.ReferenceSolutions.Hertz import (radius_and_pressure,
                                               surface_displacements,
                                               surface_stress)
    from FFTEngine import PFFTEngine
    from MPITools.Optimization import LBFGS
    from MPITools.Tools.ParallelNumpy import ParallelNumpy  # TODO: This should be explicitly from MPITools
    from mpi4py import MPI
    from PyCo.ContactMechanics import VDW82smoothMin, VDW82
    from PyCo.System import SmoothContactSystem
    from PyCo.Tools.NetCDF import NetCDFContainer

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


    class Parallel_Topography(): # Just some Temp implementation of the interface
        def __init__(self,surface,fftengine):
            self.surface = surface
            self.subdomain_resolution = fftengine.subdomain_resolution # TODO: FreeElastHS: sometimes the subdomain is empty, comp_slice ?
            self.subdomain_slice = fftengine.subdomain_slice

            self.domain_resolution = fftengine.domain_resolution
            self.resolution = self.surface.resolution

        def array(self,*args,**kwargs):
            return self.surface.array()[self.subdomain_slice]


    surface = Sphere(radius = R, resolution = surf_res ,size =surf_size)
    psurface = Parallel_Topography(surface, fftengine)

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