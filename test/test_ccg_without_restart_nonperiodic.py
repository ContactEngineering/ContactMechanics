

import numpy as np
from NuMPI.Optimization import ccg_without_restart
from NuMPI.Tools import Reduction
from SurfaceTopography import make_sphere

from ContactMechanics import FreeFFTElasticHalfSpace
from ContactMechanics.Systems import NonSmoothContactSystem
from NuMPI import MPI

import scipy.optimize as optim


def test_ccg_without_restart_free_system(comm):
    pnp = Reduction(comm)

    nx, ny = 9, 9
    sx = sy = 4.
    R = 1.
    Es = 0.75

    # MAKE REFERENCE solution in serial

    surface = make_sphere(R, (nx, ny), (sx, sy),
                          centre=(sx / 2, sy / 2),
                          kind="paraboloid", communicator=MPI.COMM_SELF)

    substrate = FreeFFTElasticHalfSpace(
        (nx, ny), young=Es,
        physical_sizes=(sx, sy), communicator=MPI.COMM_SELF)

    system = NonSmoothContactSystem(substrate, surface)

    penetration = 0.5
    lbounds = system._lbounds_from_heights(penetration)

    bnds = system._reshape_bounds(lbounds, )
    init_disp = np.zeros(substrate.nb_subdomain_grid_pts)

    bounded = init_disp < lbounds
    init_disp[bounded.filled(False)] = lbounds[bounded.filled(False)]

    res = optim.minimize(system.objective(penetration, gradient=True,),
                         init_disp,
                         method='L-BFGS-B', jac=True,
                         bounds=bnds,
                         options=dict(gtol=1e-13, ftol=1e-20))

    assert res.success
    _lbfgsb = res.x.reshape((2 * nx, 2 * ny))

    # Parallelized objective

    substrate = FreeFFTElasticHalfSpace((nx, ny), young=Es,
                                        physical_sizes=(sx, sy),
                                        communicator=comm,
                                        fft="mpi")

    surface = make_sphere(
        R, (nx, ny), (sx, sy),
        centre=(sx / 2, sy / 2),
        subdomain_locations=substrate.topography_subdomain_locations,
        nb_subdomain_grid_pts=substrate.topography_nb_subdomain_grid_pts,
        kind="paraboloid", communicator=comm)

    system = NonSmoothContactSystem(substrate, surface)
    lbounds_parallel = system._lbounds_from_heights(penetration)
    res = ccg_without_restart.constrained_conjugate_gradients(
        system.objective(penetration, gradient=True),
        # We also test that the logger and the postprocessing involved work properly in parallel
        system.hessian_product,
        init_disp[substrate.subdomain_slices].reshape(-1),
        gtol=1e-13,
        bounds=lbounds_parallel.filled().reshape(-1),
        maxiter=1000,
        communicator=comm,
    )
    assert res.success

    print(res.nit)
    _bug = res.x.reshape(substrate.nb_subdomain_grid_pts)

    if False:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(_bug[:, ny // 2], label="bug")
        ax.plot(_lbfgsb[:, ny // 2], label="lbfgsb")
        ax.legend()
        plt.show()

        fig, ax = plt.subplots()
        ax.plot(system.objective(penetration, gradient=True)(_bug)[1]
                .reshape((2 * nx, 2 * ny))[:, ny // 2], label="bug")
        ax.plot(system.objective(penetration, gradient=True)(_lbfgsb)[1]
                .reshape((2 * nx, 2 * ny))[:, ny // 2], label="lbfgsb")
        ax.legend()
        plt.show()

    assert pnp.max(abs(_bug - _lbfgsb[substrate.subdomain_slices])) < 1e-5
