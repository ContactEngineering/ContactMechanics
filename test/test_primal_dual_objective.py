from SurfaceTopography import make_sphere
import ContactMechanics as Solid
from ContactMechanics.Systems import NonSmoothContactSystem
import scipy.optimize as optim

import numpy as np
import pytest


# import matplotlib.pyplot as plt

@pytest.mark.parametrize("s", [1., 2.])
def test_primal_obj(s):
    nx, ny = 256, 256
    sx = sy = s
    R = 10.

    surface = make_sphere(R, (nx, ny), (sx, sy), kind="paraboloid")
    Es = 50.
    substrate = Solid.PeriodicFFTElasticHalfSpace((nx, ny), young=Es,
                                                  physical_sizes=(sx, sy))

    system = Solid.Systems.NonSmoothContactSystem(substrate, surface)

    offset = 0.005
    lbounds = np.zeros((nx, ny))
    bnds = system._reshape_bounds(lbounds, )
    init_gap = np.zeros((nx, ny))  # .flatten()
    disp = init_gap + surface.heights() + offset

    # res = optim.minimize(system.primal_objective(offset, gradient=True),
    #                      disp,
    #                      method='L-BFGS-B', jac=True,
    #                      bounds=bnds,
    #                      options=dict(gtol=1e-8, ftol=1e-20))

    res = system.primal_minimize_proxy(x0=disp, offset=offset,gtol=1e-8)
    assert res.success
    _bug = res.x.reshape((nx, ny))

    res = system.minimize_proxy(offset=offset, pentol=1e-7)
    assert res.success
    _ccg = system.compute_gap(res.x, offset)

    # fig, (axg, axpl, axpc) = plt.subplots(3, 1)
    #
    # plt.colorbar(axpl.pcolormesh(_lbfgsb))
    # plt.colorbar(axpc.pcolormesh(_ccg))
    # axg.plot(system.surface.positions()[0][:,0], _lbfgsb[:,ny//2],'x',
    # label='lbfgsb' )
    # axg.plot(system.surface.positions()[0][:,0], _ccg[:, ny // 2], '+',
    # label='ccg')
    # axg.legend()
    # plt.show()
    # fig.tight_layout()
    np.testing.assert_allclose(_bug, _ccg, atol=1e-6)


@pytest.mark.parametrize("s", [1., 2.])
def test_dual_obj(s):
    nx, ny = 128, 128
    sx = sy = s
    R = 10.

    surface = make_sphere(R, (nx, ny), (sx, sy), kind="paraboloid")
    Es = 50.
    substrate = Solid.PeriodicFFTElasticHalfSpace((nx, ny), young=Es,
                                                  physical_sizes=(sx, sy))

    system = Solid.Systems.NonSmoothContactSystem(substrate, surface)

    offset = 0.005
    lbounds = np.zeros((nx, ny))
    bnds = system._reshape_bounds(lbounds, )
    init_gap = np.zeros((nx, ny))
    disp = init_gap + surface.heights() + offset
    init_pressure = substrate.evaluate_force(disp)

    # res = optim.minimize(system.dual_objective(offset, gradient=True),
    #                      init_pressure,
    #                      method='L-BFGS-B', jac=True,
    #                      bounds=bnds,
    #                      options=dict(gtol=1e-8 * system.area_per_pt,
    #                                   ftol=1e-20))
    res = system.dual_minimize_proxy(x0=disp, offset=offset, gtol=1e-8)

    assert res.success
    CA_lbfgsb = res.x.reshape((nx, ny)) > 0  # Contact area
    fun = system.dual_objective(offset, gradient=True)
    gap_lbfgsb = fun(res.x)[1]
    gap_lbfgsb = gap_lbfgsb.reshape((nx, ny))

    res = system.minimize_proxy(offset=offset, pentol=1e-8)
    assert res.success

    CA_ccg = res.jac > 0  # Contact area
    # print("shape of disp_ccg  {}".format(np.shape(res.x)))
    gap_ccg = system.compute_gap(res.x, offset)

    np.testing.assert_allclose(CA_lbfgsb, CA_ccg, 1e-8)
    np.testing.assert_allclose(gap_lbfgsb, gap_ccg, atol=1e-8)


@pytest.mark.parametrize("s", (1., 2.))
def test_primal_hessian(s):
    nx = 64
    ny = 32

    sx = sy = s
    R = 10.
    Es = 50.

    substrate = Solid.PeriodicFFTElasticHalfSpace((nx, ny), young=Es,
                                                  physical_sizes=(sx, sy))

    topography = make_sphere(R, (nx, ny), (sx, sy), kind="paraboloid")

    system = NonSmoothContactSystem(substrate=substrate, surface=topography)

    obj = system.primal_objective(0, True)

    gaps = np.random.random(size=(nx, ny))
    dgaps = np.random.random(size=(nx, ny))

    _, grad = obj(gaps)

    h = 1.
    _, grad_d = obj(gaps + h * dgaps)

    dgrad = grad_d - grad

    dgrad_from_hess = system.primal_hessian_product(h * dgaps)

    np.testing.assert_allclose(dgrad_from_hess, dgrad)

@pytest.mark.parametrize("s", (1., 2.))
def test_dual_hessian(s):
    nx = 64
    ny = 32

    sx = sy = s
    R = 10.
    Es = 50.

    substrate = Solid.PeriodicFFTElasticHalfSpace((nx, ny), young=Es,
                                                  physical_sizes=(sx, sy))

    topography = make_sphere(R, (nx, ny), (sx, sy), kind="paraboloid")

    system = NonSmoothContactSystem(substrate=substrate, surface=topography)

    obj = system.dual_objective(0, True)

    gaps = np.random.random(size=(nx, ny))
    dgaps = np.random.random(size=(nx, ny))

    _, grad = obj(gaps)

    h = 1.
    _, grad_d = obj(gaps + h * dgaps)

    dgrad = grad_d - grad

    dgrad_from_hess = system.dual_hessian_product(h * dgaps)

    np.testing.assert_allclose(dgrad_from_hess, dgrad)
