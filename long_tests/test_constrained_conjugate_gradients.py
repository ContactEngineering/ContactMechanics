from SurfaceTopography import make_sphere
import ContactMechanics as Solid
import numpy as np
import scipy.optimize as optim
from NuMPI.Optimization import ccg_without_restart, ccg_with_restart
import pytest
from NuMPI import MPI

pytestmark = pytest.mark.skipif(MPI.COMM_WORLD.Get_size() > 1,
                                reason="tests only serial funcionalities, "
                                       "please execute with pytest")

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
    disp = np.zeros((nx, ny))
    init_gap = disp - surface.heights() - offset

    # ####################POLONSKY-KEER##############################
    res = ccg_with_restart.constrained_conjugate_gradients(
        system.primal_objective(offset, gradient=True),
        system.primal_hessian_product, x0=init_gap, gtol=gtol)

    assert res.success
    polonsky_gap = res.x.reshape((nx, ny))

    # ####################BUGNICOURT###################################
    res = ccg_without_restart.constrained_conjugate_gradients(system.primal_objective
                                                        (offset,
                                                         gradient=True),
                                                        system.primal_hessian_product,
                                                        x0=init_gap,
                                                        mean_val=None,
                                                        gtol=gtol)
    assert res.success

    bugnicourt_gap = res.x.reshape((nx, ny))

    # #####################LBFGSB#####################################
    res = optim.minimize(system.primal_objective(offset, gradient=True),
                         init_gap,
                         method='L-BFGS-B', jac=True,
                         bounds=bnds,
                         options=dict(gtol=gtol, ftol=1e-20))

    assert res.success
    lbfgsb_gap = res.x.reshape((nx, ny))

    np.testing.assert_allclose(polonsky_gap, bugnicourt_gap, atol=1e-3)
    np.testing.assert_allclose(polonsky_gap, lbfgsb_gap, atol=1e-3)
    np.testing.assert_allclose(lbfgsb_gap, bugnicourt_gap, atol=1e-3)

    # ##########TEST MEAN VALUES#######################################
    mean_val = np.mean(lbfgsb_gap)
    # ####################POLONSKY-KEER##############################
    res = ccg_with_restart.constrained_conjugate_gradients(
        system.primal_objective(offset, gradient=True),
        system.primal_hessian_product, init_gap, gtol=gtol,
        mean_value=mean_val)

    assert res.success
    polonsky_gap_mean_cons = res.x.reshape((nx, ny))

    # ####################BUGNICOURT###################################
    ccg_without_restart.constrained_conjugate_gradients(system.primal_objective
                                                  (offset, gradient=True),
                                                  system.
                                                  primal_hessian_product,
                                                  x0=init_gap,
                                                  mean_val=mean_val,
                                                  gtol=gtol
                                                  )
    assert res.success

    bugnicourt_gap_mean_cons = res.x.reshape((nx, ny))

    np.testing.assert_allclose(polonsky_gap_mean_cons, lbfgsb_gap, atol=1e-3)
    np.testing.assert_allclose(bugnicourt_gap_mean_cons, lbfgsb_gap, atol=1e-3)
    np.testing.assert_allclose(lbfgsb_gap, bugnicourt_gap, atol=1e-3)
    np.testing.assert_allclose(lbfgsb_gap, bugnicourt_gap_mean_cons, atol=1e-3)


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
    lbfgsb_force = res.x.reshape((nx, ny))
    CA_lbfgsb = res.x.reshape((nx, ny)) > 0  # Contact area
    fun = system.dual_objective(offset, gradient=True)
    gap_lbfgsb = fun(res.x)[1]
    gap_lbfgsb = gap_lbfgsb.reshape((nx, ny))

    # ###################BUGNICOURT########################################
    ccg_without_restart.constrained_conjugate_gradients(
        system.dual_objective(offset, gradient=True),
        system.
            dual_hessian_product,
        init_pressure,
        mean_val=None, gtol=gtol)
    assert res.success

    bugnicourt_force = res.x.reshape((nx, ny))
    CA_bugnicourt = res.x.reshape((nx, ny)) > 0  # Contact area
    gap_bugnicourt = fun(res.x)[1]
    gap_bugnicourt = gap_bugnicourt.reshape((nx, ny))

    # # ##################POLONSKY-KEER#####################################
    res = ccg_with_restart.constrained_conjugate_gradients(
        system.dual_objective(offset, gradient=True),
        system.dual_hessian_product, init_pressure, gtol=gtol)
    assert res.success

    polonsky_force = res.x
    CA_polonsky = res.x.reshape((nx, ny)) > 0  # Contact area
    gap_polonsky = fun(res.x)[1]
    gap_polonsky = gap_polonsky.reshape((nx, ny))

    np.testing.assert_allclose(gap_lbfgsb, gap_polonsky, atol=1e-3)
    np.testing.assert_allclose(gap_lbfgsb, gap_bugnicourt, atol=1e-3)
    np.testing.assert_allclose(gap_bugnicourt, gap_polonsky, atol=1e-3)
    np.testing.assert_allclose(lbfgsb_force, bugnicourt_force, atol=1e-3)
    np.testing.assert_allclose(lbfgsb_force,
                               polonsky_force.reshape(lbfgsb_force.shape),
                               atol=1e-3)
    np.testing.assert_allclose(polonsky_force.reshape(lbfgsb_force.shape),
                               bugnicourt_force, atol=1e-3)

    # ##########TEST MEAN VALUES#######################################
    mean_val = np.mean(lbfgsb_force)
    # print('mean {}'.format(mean_val))
    # ####################POLONSKY-KEER##############################
    res = ccg_with_restart.constrained_conjugate_gradients(
        system.dual_objective(offset, gradient=True),
        system.dual_hessian_product, init_pressure, gtol=gtol,
        mean_value=mean_val)

    assert res.success
    polonsky_force_mean_cons = res.x.reshape((nx, ny))

    # # ####################BUGNICOURT###################################
    ccg_without_restart.constrained_conjugate_gradients(
        system.dual_objective(offset, gradient=True),
        system.
            dual_hessian_product,
        init_pressure,
        mean_val=mean_val,
        gtol=gtol)
    assert res.success

    bugnicourt_force_mean_cons = res.x.reshape((nx, ny))

    np.testing.assert_allclose(polonsky_force_mean_cons, lbfgsb_force,
                               atol=1e-3)
    np.testing.assert_allclose(bugnicourt_force_mean_cons, lbfgsb_force,
                               atol=1e-3)
