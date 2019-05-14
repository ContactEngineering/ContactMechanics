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
    import pytest
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
    from NuMPI.Optimization import LBFGS
    from NuMPI.Tools.Reduction import Reduction  # TODO: This should be explicitly from NuMPI
    from mpi4py import MPI
    from PyCo.ContactMechanics import VDW82smoothMin, VDW82
    from PyCo.System import SmoothContactSystem
    from PyCo.Tools.NetCDF import NetCDFContainer
    from PyCo.Topography import Topography

    import PyCo.ReferenceSolutions.DMT as DMT
    import PyCo.ReferenceSolutions.JKR as JKR
    import PyCo.ReferenceSolutions.MaugisDugdale as MD
    from scipy.optimize import minimize_scalar


except ImportError as err:
    import sys
    print(err)
    sys.exit(-1)

_toplot =True

@pytest.mark.skip("is very slow, call it explicitely")
def test_smoothsphere(maxcomm, fftengine_class): # TODO problem: difficult to compare contact_area with MD Model,
    """
    This test needs a lot of computational effort
    Parameters
    ----------
    maxcomm

    Returns
    -------

    """
    comm = maxcomm
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

    nx, ny = 512, 512
    sx = 21.0

    z0 = 0.05 # needed to get small tolerance, but very very slow

    fftengine = fftengine_class((2*nx, 2*ny), comm=comm)
    pnp =Reduction(comm=comm)

    # the "Min" part of the potential (linear for small z) is needed for the LBFGS without bounds
    inter = VDW82smoothMin(w * z0 ** 8 / 3, 16 * np.pi * w * z0 ** 2, gamma=w, pnp = pnp)

    # Parallel Topography Patch

    substrate = FreeFFTElasticHalfSpace((nx,ny), young=E_s, size=(sx,sx), fftengine=fftengine, pnp=pnp)
    print(substrate._comp_resolution)
    print(fftengine.domain_resolution)


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

    offsets =  np.flip(np.linspace(r_s / 100 - z0, r_s/50 - z0, 11))

    force = np.zeros_like(offsets)

    nsteps = len(offsets)

    disp0 = ext_surface.heights() + offsets[0]
    disp0 = np.where(disp0 > 0, disp0, 0)
    disp0 = system.shape_minimisation_input(disp0)
    normal_force = []
    area = []
    for i in range(nsteps):
        result = LBFGS(system.objective(offsets[i], gradient=True), disp0, jac=True, pnp=pnp, gtol=1e-6 * abs(w/z0))

        #result = system.minimize_proxy(offsets[i], disp0=None,method = LBFGS,options=dict(gtol = 1e-3, maxiter =100,maxls=10))

        u = result.x
        u.shape = ext_surface.subdomain_resolution
        f = substrate.evaluate_force(u)
        converged = result.success
        assert converged

        gap = system.compute_gap(u, offsets[i])

        normal_force += [-pnp.sum(f)]
        area += [(pnp.sum(gap < inter.r_infl)) * system.area_per_pt]

    normal_force = np.array(normal_force)
    area = np.array(area)

    # fits the best cohesive stress for the MD-model
    opt = minimize_scalar(lambda x: ((MD.load_and_displacement(
         np.sqrt(area / np.pi), r_s, E_s, w, x)[0] - normal_force) ** 2).sum(),
                           bracket=(0.1 * w / z0, 1.02* w / z0))
    #                                             ^- max attractive stress is  approx
    #                                                1.02 * w /z0 in the potential
    cohesive_stress = opt.x
    print("cohesive_stress: {}".format(cohesive_stress))
    print("potential: max attractive stress: {}".format(1.02*w/z0))
    #
    residual = np.sqrt(((MD.load_and_displacement(np.sqrt(area / np.pi), r_s,
                                                  E_s, w, cohesive_stress)[
                             0] - normal_force) ** 2).mean())
    print("residual {}".format(residual))
    #assert residual < 1, "residual = {} >=01".format(residual)

    toPlot = comm.Get_rank() == 0 and _toplot
    if toPlot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        for t_cohesive_stress in [0.1 * w / z0, 1.03 * w / z0]:
            ax.plot(area,
                    MD.load_and_displacement(np.sqrt(area / np.pi), r_s, E_s, w,
                                             t_cohesive_stress)[0],
                    label="analytical cohesive_stress {}".format(
                        t_cohesive_stress))
        ax.plot(area,
                MD.load_and_displacement(np.sqrt(area / np.pi), r_s, E_s, w,
                                         cohesive_stress)[0],
                label="analytical cohesive_stress {}".format(
                    cohesive_stress))
        ax.plot(area, normal_force, label="numerical")

        ax.set_xlabel("area")
        ax.set_ylabel("normal_force")
        ax.grid(True)
        ax.legend()

        fig.tight_layout()
        fig.savefig("test_smoothsphere_area_force.png")
        fig2, ax2 = plt.subplots()

        ax2.plot(offsets, normal_force)
        fig2.savefig("test_smoothsphere_penetration_force.png")



if __name__ == "__main__":
    from mpi4py import MPI
    import matplotlib.pyplot as plt
    _toplot = True
    test_smoothsphere(MPI.COMM_WORLD)
    plt.show(block=True)