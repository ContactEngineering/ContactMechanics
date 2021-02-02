from SurfaceTopography import make_sphere
import ContactMechanics as Solid
import numpy as np
import scipy.optimize as optim
import sys
# sys.path.insert(1, '/home/sindhu/Downloads/Thesis/code/SindhuThesis')
# from optimiser import generic_cg_polonsky, bugnicourt_cg
from NuMPI.Optimization import bugnicourt_cg, polonsky_keer


def test_primal_obj():
    nx = ny = 1024
    sx, sy = 1., 1.
    R = 10.

    gtol = 1e-6

    surface = make_sphere(R, (nx, ny), (sx, sy), kind="paraboloid")
    Es = 50.
    substrate = Solid.PeriodicFFTElasticHalfSpace((nx, ny), young=Es,
                                                  physical_sizes=(sx, sy))

    system = Solid.Systems.NonSmoothContactSystem(substrate, surface)

    offset = 0.005
    lbounds = np.zeros((nx, ny))
    bnds = system._reshape_bounds(lbounds, )
    init_gap = np.zeros((nx, ny))  # .flatten()
    # disp = init_gap + surface.heights() + offset
    disp = np.zeros((nx, ny))
    init_gap = disp - surface.heights() - offset

    # ####################POLONSKY-KEER##############################
    res = polonsky_keer.min_cg(system.primal_objective(offset, gradient=True),
                               system.primal_hessian_product,
                               x0=init_gap, gtol=gtol)

    assert res.success
    polonsky = res.x.reshape((nx, ny))

    # ####################BUGNICOURT###################################
    res = bugnicourt_cg.constrained_conjugate_gradients(system.primal_objective
                                                        (offset,
                                                         gradient=True),
                                                        system.primal_hessian_product,
                                                        x0=init_gap,
                                                        mean_val=None,
                                                        gtol=gtol)
    assert res.success

    bugnicourt = res.x.reshape((nx, ny))

    # #####################LBFGSB#####################################
    res = optim.minimize(system.primal_objective(offset, gradient=True),
                         init_gap,
                         method='L-BFGS-B', jac=True,
                         bounds=bnds,
                         options=dict(gtol=gtol, ftol=1e-20))

    assert res.success
    _lbfgsb = res.x.reshape((nx, ny))

    np.testing.assert_allclose(polonsky, bugnicourt, atol=1e-3)
    np.testing.assert_allclose(polonsky, _lbfgsb, atol=1e-3)
    np.testing.assert_allclose(_lbfgsb, bugnicourt, atol=1e-3)

    # ##########TEST MEAN VALUES#######################################
    mean_val = np.mean(_lbfgsb)
    # ####################POLONSKY-KEER##############################
    res = polonsky_keer.min_cg(
        system.primal_objective(offset, gradient=True),
        system.primal_hessian_product,
        init_gap, mean_value=mean_val, gtol=gtol)

    assert res.success
    polonsky_mean = res.x.reshape((nx, ny))

    # ####################BUGNICOURT###################################
    bugnicourt_cg.constrained_conjugate_gradients(system.primal_objective
                                                  (offset, gradient=True),
                                                  system.
                                                  primal_hessian_product,
                                                  x0=init_gap,
                                                  mean_val=mean_val,
                                                  gtol=gtol
                                                  )
    assert res.success

    bugnicourt_mean = res.x.reshape((nx, ny))

    np.testing.assert_allclose(polonsky_mean, _lbfgsb, atol=1e-3)
    np.testing.assert_allclose(bugnicourt_mean, _lbfgsb, atol=1e-3)
    np.testing.assert_allclose(_lbfgsb, bugnicourt, atol=1e-3)
    np.testing.assert_allclose(_lbfgsb, bugnicourt_mean, atol=1e-3)


def test_dual_obj():
    nx = ny = 1024
    sx, sy = 1., 1.
    R = 10.

    gtol = 1e-7

    surface = make_sphere(R, (nx, ny), (sx, sy), kind="paraboloid")
    Es = 50.
    substrate = Solid.PeriodicFFTElasticHalfSpace((nx, ny), young=Es,
                                                  physical_sizes=(sx, sy))
    substrate_2 = Solid.PeriodicFFTElasticHalfSpace((nx, ny), young=Es,
                                                    physical_sizes=(sx, sy),
                                                    stiffness_q0=0.0)

    system = Solid.Systems.NonSmoothContactSystem(substrate, surface)
    system_2 = Solid.Systems.NonSmoothContactSystem(substrate_2, surface, )

    print("max height {}".format(np.max(abs(system.surface.heights()))))

    offset = 0.005
    lbounds = np.zeros((nx, ny))
    bnds = system._reshape_bounds(lbounds, )
    init_gap = np.zeros((nx, ny))
    disp = init_gap + surface.heights() + offset
    init_pressure = substrate.evaluate_force(disp)

    # ####################LBFGSB########################################
    res = optim.minimize(system.dual_objective(offset, gradient=True),
                         init_pressure,
                         method='L-BFGS-B', jac=True,
                         bounds=bnds,
                         options=dict(gtol=gtol, ftol=1e-20))

    print(res.message, res.nfev)
    assert res.success
    CA_lbfgsb = res.x.reshape((nx, ny)) > 0  # Contact area
    print("CA lbfgsb {}".format(CA_lbfgsb.sum() / (nx * ny)))
    # print(CA_lbfgsb / (nx * ny))
    _lbfgsb = res.x.reshape((nx, ny))
    fun = system.dual_objective(offset, gradient=True)
    gap_lbfgsb = fun(res.x)[1]
    gap_lbfgsb = gap_lbfgsb.reshape((nx, ny))

    # ###################BUGNICOURT########################################
    bugnicourt_cg.constrained_conjugate_gradients(system.dual_objective
                                                  (offset, gradient=True),
                                                  system.
                                                  dual_hessian_product,
                                                  init_pressure,
                                                  mean_val=None, gtol=gtol)
    assert res.success

    bugnicourt = res.x.reshape((nx, ny))
    CA_bugnicourt = res.x.reshape((nx, ny)) > 0  # Contact area
    gap_bugnicourt = fun(res.x)[1]
    gap_bugnicourt = gap_bugnicourt.reshape((nx, ny))
    #
    # # ##################POLONSKY-KEER#####################################
    res = polonsky_keer.min_cg(
        system.dual_objective(offset, gradient=True),
        system.dual_hessian_product,
        init_pressure, gtol=gtol)
    assert res.success

    CA_polonsky = res.x.reshape((nx, ny)) > 0  # Contact area
    gap_polonsky = fun(res.x)[1]
    gap_polonsky = gap_polonsky.reshape((nx, ny))

    np.testing.assert_allclose(gap_lbfgsb, gap_polonsky, atol=1e-3)
    np.testing.assert_allclose(gap_lbfgsb, gap_bugnicourt, atol=1e-3)
    np.testing.assert_allclose(gap_bugnicourt, gap_polonsky, atol=1e-3)

    # ##########TEST MEAN VALUES#######################################
    mean_val = np.mean(_lbfgsb)
    print('mean {}'.format(mean_val))
    # ####################POLONSKY-KEER##############################
    res = polonsky_keer.min_cg(
        system.dual_objective(offset, gradient=True),
        system.dual_hessian_product,
        init_pressure, mean_value=mean_val, gtol=gtol)

    assert res.success
    polonsky_mean = res.x.reshape((nx, ny))

    # # ####################BUGNICOURT###################################
    bugnicourt_cg.constrained_conjugate_gradients(system.dual_objective
                                                  (offset, gradient=True),
                                                  system.
                                                  dual_hessian_product,
                                                  init_pressure,
                                                  mean_val=mean_val,
                                                  gtol=gtol)
    assert res.success

    bugnicourt_mean = res.x.reshape((nx, ny))

    np.testing.assert_allclose(polonsky_mean, _lbfgsb, atol=1e-3)
    np.testing.assert_allclose(bugnicourt_mean, _lbfgsb, atol=1e-3)
