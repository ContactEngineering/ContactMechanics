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

    offset = 0.05

    init_gap = np.zeros((nx, ny))  # .flatten()
    disp = init_gap + surface.heights() + offset

    res = system.primal_minimize_proxy(offset, init_gap=disp,
                                       solver='ccg_without_restart', )

    assert res.success
    CA_bug = res.x.reshape((nx, ny)) == 0  # Contact area
    force_bug = res.jac
    gap_bug = res.x
    gap_bug = gap_bug.reshape((nx, ny))

    res = system.primal_minimize_proxy(offset, init_gap=disp,
                                       solver='ccg_with_restart', )

    assert res.success
    CA_pk = res.x.reshape((nx, ny)) == 0  # Contact are
    force_pk = res.jac
    gap_pk = res.x
    gap_pk = gap_pk.reshape((nx, ny))

    res = system.primal_minimize_proxy(offset, init_gap=disp,
                                       solver='l-bfgs-b', )
    assert res.success
    CA_lbfgsb = res.x.reshape((nx, ny)) == 0  # Contact area
    force_lbfgsb = res.jac
    gap_lbfgsb = res.x
    gap_lbfgsb = gap_lbfgsb.reshape((nx, ny))

    np.testing.assert_allclose(CA_bug, CA_lbfgsb, atol=1e-3)
    np.testing.assert_allclose(gap_bug, gap_lbfgsb, atol=1e-3)
    np.testing.assert_allclose(force_bug, force_lbfgsb, atol=1e-3)
    np.testing.assert_allclose(CA_pk, CA_lbfgsb, atol=1e-3)
    np.testing.assert_allclose(gap_pk, gap_lbfgsb, atol=1e-3)
    np.testing.assert_allclose(force_pk, force_lbfgsb, atol=1e-3)


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

    init_gap = np.zeros((nx, ny))
    disp = init_gap + surface.heights() + offset
    disp[disp < 0] = 0
    init_force = substrate.evaluate_force(disp)

    res = system.dual_minimize_proxy(offset, init_force=init_force,
                                     solver='ccg_without_restart', )

    assert res.success
    CA_bug = res.x.reshape((nx, ny)) > 0  # Contact area
    force_bug = res.x
    fun = system.dual_objective(offset, gradient=True)
    gap_bug = fun(res.x)[1]
    gap_bug = gap_bug.reshape((nx, ny))

    res = system.dual_minimize_proxy(offset, init_force=init_force,
                                     solver='ccg_with_restart', )

    assert res.success, res.message
    CA_pk = res.x.reshape((nx, ny)) > 0  # Contact are
    force_pk = res.x
    fun = system.dual_objective(offset, gradient=True)
    gap_pk = fun(res.x)[1]
    gap_pk = gap_pk.reshape((nx, ny))

    res = system.dual_minimize_proxy(offset, init_force=init_force,
                                     solver='l-bfgs-b', )

    assert res.success, res.message
    CA_lbfgsb = res.x.reshape((nx, ny)) > 0  # Contact area
    force_lbfgsb = res.x
    fun = system.dual_objective(offset, gradient=True)
    gap_lbfgsb = fun(res.x)[1]
    gap_lbfgsb = gap_lbfgsb.reshape((nx, ny))

    np.testing.assert_allclose(CA_bug, CA_lbfgsb, atol=1e-5)
    np.testing.assert_allclose(gap_bug, gap_lbfgsb, atol=1e-5)
    np.testing.assert_allclose(force_bug, force_lbfgsb, atol=1e-5)
    np.testing.assert_allclose(CA_pk, CA_lbfgsb, atol=1e-5)
    np.testing.assert_allclose(gap_pk, gap_lbfgsb, atol=1e-5)
    np.testing.assert_allclose(force_pk, force_lbfgsb, atol=1e-5)


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

    _, grad = obj(gaps.reshape(-1))

    h = 1.
    _, grad_d = obj((gaps + h * dgaps).reshape(-1))

    dgrad = grad_d - grad

    dgrad_from_hess = system.primal_hessian_product((h * dgaps).reshape(-1))

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

    obj = system.dual_objective(0)

    gaps = np.random.random(size=(nx, ny))
    dgaps = np.random.random(size=(nx, ny))

    _, grad = obj(gaps)

    h = 1.
    _, grad_d = obj(gaps + h * dgaps)

    dgrad = grad_d - grad

    dgrad_from_hess = system.dual_hessian_product(h * dgaps)

    np.testing.assert_allclose(dgrad_from_hess.reshape(dgrad.shape), dgrad)


@pytest.mark.parametrize("s", [1., 2.])
def test_scipy_friendly_interface_nonperiodic(s):
    # TODO: there is an old bug in the nonsmoothcontactsystem objective
    nx, ny = 32, 32
    sx = sy = s
    R = 10.

    surface = make_sphere(R, (nx, ny), (sx, sy),
                          centre=(sx / 2, sy / 2),
                          kind="paraboloid")
    padded_surface = make_sphere(R, (2 * nx, 2 * ny), (2 * sx, 2 * sy),
                                 centre=(sx / 2, sy / 2),
                                 kind="paraboloid")
    Es = 50.
    substrate = Solid.FreeFFTElasticHalfSpace((nx, ny), young=Es,
                                              physical_sizes=(sx, sy))

    system = Solid.Systems.NonSmoothContactSystem(substrate, surface)

    penetration = 0.005
    lbounds = system._lbounds_from_heights(penetration)

    bnds = system._reshape_bounds(lbounds, )
    init_disp = np.ones(substrate.nb_subdomain_grid_pts)  # .flatten()
    init_gap = init_disp - padded_surface.heights() - penetration

    res = optim.minimize(system.objective(penetration, gradient=True),
                         init_gap,
                         method='L-BFGS-B', jac=True,
                         bounds=bnds,
                         options=dict(gtol=1e-8, ftol=1e-20))

    assert res.success
    _lbfgsb = res.x.reshape((2 * nx, 2 * ny))

    res = system.minimize_proxy(offset=penetration, pentol=1e-7)
    assert res.success
    _ccg = res.x

    np.testing.assert_allclose(_lbfgsb, _ccg, atol=1e-6)
