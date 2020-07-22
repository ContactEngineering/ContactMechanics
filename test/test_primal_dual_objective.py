from SurfaceTopography import make_sphere
import ContactMechanics as Solid
import scipy.optimize as optim

import numpy as np

import matplotlib.pyplot as plt


def test_primal_obj():
    nx, ny = 256, 256
    sx, sy = 1., 1.
    R = 10.

    surface = make_sphere(R, (nx, ny), (sx, sy), kind="paraboloid")
    Es = 50.
    substrate = Solid.PeriodicFFTElasticHalfSpace((nx, ny), young=Es,
                                                  physical_sizes=(sx, sy))

    system = Solid.Systems.NonSmoothContactSystem(substrate, surface)

    gtol = 1e-8
    offset = 0.005
    lbounds = np.zeros((nx, ny))
    bnds = system._reshape_bounds(lbounds, )
    init_gap = np.zeros((nx, ny))  # .flatten()
    disp = init_gap + surface.heights() + offset

    res = optim.minimize(system.primal_objective(offset, gradient=True),
                         disp,
                         method='L-BFGS-B', jac=True,
                         bounds=bnds,
                         options=dict(gtol=1e-8, ftol=1e-20))

    assert res.success
    _lbfgsb = res.x.reshape((nx, ny))

    res = system.minimize_proxy(offset=offset, pentol=1e-7)
    assert res.success
    _ccg = system.compute_gap(res.x, offset)

    # fig, (axg, axpl, axpc) = plt.subplots(3, 1)
    #
    # plt.colorbar(axpl.pcolormesh(_lbfgsb))
    # plt.colorbar(axpc.pcolormesh(_ccg))
    # axg.plot(system.surface.positions()[0][:,0], _lbfgsb[:,ny//2],'x', label='lbfgsb' )
    # axg.plot(system.surface.positions()[0][:,0], _ccg[:, ny // 2], '+',label='ccg')
    # axg.legend()
    # plt.show()
    # fig.tight_layout()
    np.testing.assert_allclose(_lbfgsb, _ccg, atol=1e-6)


def test_dual_obj():
    nx, ny = 128, 128
    sx, sy = 1., 1.
    R = 10.

    surface = make_sphere(R, (nx, ny), (sx, sy), kind="paraboloid")
    Es = 50.
    substrate = Solid.PeriodicFFTElasticHalfSpace((nx, ny), young=Es,
                                                  physical_sizes=(sx, sy))

    system = Solid.Systems.NonSmoothContactSystem(substrate, surface)

    gtol = 1e-8
    offset = 0.05
    lbounds = np.zeros((nx, ny))
    bnds = system._reshape_bounds(lbounds, )
    init_gap = np.zeros((nx, ny))  # .flatten()
    disp = init_gap + surface.heights() + offset
    init_pressure = substrate.evaluate_force(disp)

    res = optim.minimize(system.dual_objective(offset, gradient=True),
                         init_pressure,
                         method='L-BFGS-B', jac=True,
                         bounds=bnds,
                         options=dict(gtol=1e-8, ftol=1e-20))
    assert res.success
    CA_lbfgsb = res.x.reshape((nx, ny)) > 0  # Contact area

    res = system.minimize_proxy(offset=offset)
    assert res.success

    CA_ccg = res.jac > 0  # Contact area
    np.testing.assert_allclose(CA_lbfgsb, CA_ccg, 1e-8)
