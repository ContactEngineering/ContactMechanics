from SurfaceTopography import make_sphere
import ContactMechanics as Solid
from NuMPI.Optimization import generic_cg_polonsky, bugnicourt_cg
import numpy as np
import scipy.optimize as optim
import pytest
from NuMPI import MPI

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="NuMPI CGs not MPI yet")


@pytest.mark.parametrize('_solver', ['generic_cg_polonsky', 'bugnicourt_cg'])
def test_primal_obj(_solver):
    nx, ny = 128, 128
    sx, sy = 1., 1.
    R = 10.

    gtol = 1e-8

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

    # #####################LBFGSB#####################################
    res = optim.minimize(system.primal_objective(offset, gradient=True),
                         disp,
                         method='L-BFGS-B', jac=True,
                         bounds=bnds,
                         options=dict(gtol=1e-8, ftol=1e-20))

    assert res.success
    _lbfgsb = res.x.reshape((nx, ny))

    if _solver == 'generic_cg_polonsky':
        # ####################POLONSKY-KEER##############################
        res = generic_cg_polonsky.min_cg(
            system.primal_objective(offset, gradient=True),
            system.primal_hessian_product,
            disp, polonskykeer=True, gtol=gtol)

        assert res.success
        polonsky = res.x.reshape((nx, ny))

        # ####################BUGNICOURT###################################
        res = generic_cg_polonsky.min_cg(
            system.primal_objective(offset, gradient=True),
            system.primal_hessian_product,
            disp, bugnicourt=True, gtol=gtol)
        assert res.success

        bugnicourt = res.x.reshape((nx, ny))

        np.testing.assert_allclose(polonsky, bugnicourt, atol=1e-3)
        np.testing.assert_allclose(polonsky, _lbfgsb, atol=1e-3)
        np.testing.assert_allclose(_lbfgsb, bugnicourt, atol=1e-3)

        # ##########TEST MEAN VALUES#######################################
        mean_val = np.mean(_lbfgsb)
        # disp = _lbfgsb
        # ####################POLONSKY-KEER##############################
        res = generic_cg_polonsky.min_cg(
            system.primal_objective(offset, gradient=True),
            system.primal_hessian_product,
            disp, polonskykeer=True, mean_value=mean_val, gtol=gtol)

        assert res.success
        polonsky_mean = res.x.reshape((nx, ny))

        # ####################BUGNICOURT###################################
        res = generic_cg_polonsky.min_cg(
            system.primal_objective(offset, gradient=True),
            system.primal_hessian_product,
            disp, bugnicourt=True, mean_value=mean_val, gtol=gtol)
        assert res.success

        bugnicourt_mean = res.x.reshape((nx, ny))

        np.testing.assert_allclose(polonsky_mean, _lbfgsb, atol=1e-3)
        np.testing.assert_allclose(bugnicourt_mean, _lbfgsb, atol=1e-3)

    elif _solver == 'bugnicourt_cg':
        bugnicourt_cg.constrained_conjugate_gradients(system.primal_objective
                                                      (offset, gradient=True),
                                                      system.
                                                      primal_hessian_product,
                                                      x0=disp,
                                                      mean_val=None, gtol=1e-8)

        bugnicourt = res.x.reshape((nx, ny))

        np.testing.assert_allclose(_lbfgsb, bugnicourt, atol=1e-3)

        mean_val = np.mean(_lbfgsb)

        bugnicourt_cg.constrained_conjugate_gradients(system.primal_objective
                                                      (offset, gradient=True),
                                                      system.
                                                      primal_hessian_product,
                                                      x0=disp,
                                                      mean_val=mean_val,
                                                      gtol=1e-8
                                                      )

        bugnicourt_mean = res.x.reshape((nx, ny))

        np.testing.assert_allclose(_lbfgsb, bugnicourt_mean, atol=1e-3)
    else:
        assert False


@pytest.mark.parametrize('_solver', ['generic_cg_polonsky', 'bugnicourt_cg'])
def test_dual_obj(_solver):
    nx, ny = 128, 128
    sx, sy = 1., 1.
    R = 10.

    gtol = 1e-8

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

    # ####################LBFGSB########################################
    res = optim.minimize(system.dual_objective(offset, gradient=True),
                         init_pressure,
                         method='L-BFGS-B', jac=True,
                         bounds=bnds,
                         options=dict(gtol=1e-8, ftol=1e-20))
    assert res.success
    CA_lbfgsb = res.x.reshape((nx, ny)) > 0  # Contact area
    press_lbfgsb = res.x.reshape((nx, ny))
    _lbfgsb = res.x.reshape((nx, ny))
    fun = system.dual_objective(offset, gradient=True)
    gap_lbfgsb = fun(res.x)[1]
    gap_lbfgsb = gap_lbfgsb.reshape((nx, ny))

    if _solver == 'generic_cg_polonsky':
        # ###################BUGNICOURT########################################
        res = generic_cg_polonsky.min_cg(
            system.dual_objective(offset, gradient=True),
            system.dual_hessian_product,
            init_pressure, bugnicourt=True, gtol=gtol)
        assert res.success

        CA_bugnicourt = res.x.reshape((nx, ny)) > 0  # Contact area
        press_bugnicourt = res.x.reshape((nx, ny))
        gap_bugnicourt = fun(res.x)[1]
        gap_bugnicourt = gap_bugnicourt.reshape((nx, ny))

        # ##################POLONSKY-KEER#####################################
        res = generic_cg_polonsky.min_cg(
            system.dual_objective(offset, gradient=True),
            system.dual_hessian_product,
            init_pressure, polonskykeer=True, gtol=gtol)
        assert res.success

        CA_polonsky = res.x.reshape((nx, ny)) > 0  # Contact area
        press_pol = res.x.reshape((nx, ny))
        gap_polonsky = fun(res.x)[1]
        gap_polonsky = gap_polonsky.reshape((nx, ny))

        np.testing.assert_allclose(CA_lbfgsb, CA_polonsky, atol=1e-3)
        np.testing.assert_allclose(gap_lbfgsb, gap_polonsky, atol=1e-3)
        np.testing.assert_allclose(CA_lbfgsb, CA_bugnicourt, atol=1e-3)
        np.testing.assert_allclose(gap_lbfgsb, gap_bugnicourt, atol=1e-3)
        np.testing.assert_allclose(CA_bugnicourt, CA_polonsky, atol=1e-3)
        np.testing.assert_allclose(gap_bugnicourt, gap_polonsky, atol=1e-3)

        np.testing.assert_allclose(press_bugnicourt, press_pol, atol=1e-3)
        np.testing.assert_allclose(press_bugnicourt, press_lbfgsb, atol=1e-3)
        np.testing.assert_allclose(press_lbfgsb, press_pol, atol=1e-3)

        # ##########TEST MEAN VALUES#######################################
        mean_val = np.mean(_lbfgsb)
        # print('mean {}'.format(mean_val))
        # ####################POLONSKY-KEER##############################
        res = generic_cg_polonsky.min_cg(
            system.dual_objective(offset, gradient=True),
            system.dual_hessian_product,
            init_pressure, polonskykeer=True, mean_value=mean_val, gtol=gtol)

        assert res.success
        polonsky_mean = res.x.reshape((nx, ny))

        # # ####################BUGNICOURT###################################
        res = generic_cg_polonsky.min_cg(
            system.dual_objective(offset, gradient=True),
            system.dual_hessian_product,
            init_pressure, mean_value=mean_val, gtol=gtol, bugnicourt=True,
            residual_plot=False, maxiter=5000)
        assert res.success

        bugnicourt_mean = res.x.reshape((nx, ny))
        print(bugnicourt_mean)

        np.testing.assert_allclose(polonsky_mean, _lbfgsb, atol=1e-3)
        np.testing.assert_allclose(bugnicourt_mean, _lbfgsb, atol=1e-3)

    elif _solver == 'bugnicourt_cg':
        bugnicourt_cg.constrained_conjugate_gradients(system.dual_objective
                                                      (offset, gradient=True),
                                                      system.
                                                      dual_hessian_product,
                                                      init_pressure,
                                                      mean_val=None, gtol=1e-8)

        bugnicourt = res.x.reshape((nx, ny))

        np.testing.assert_allclose(_lbfgsb, bugnicourt, atol=1e-3)

        mean_val = np.mean(_lbfgsb)

        bugnicourt_cg.constrained_conjugate_gradients(system.dual_objective
                                                      (offset, gradient=True),
                                                      system.
                                                      dual_hessian_product,
                                                      init_pressure,
                                                      mean_val=mean_val,
                                                      gtol=1e-8)

        bugnicourt_mean = res.x.reshape((nx, ny))

        np.testing.assert_allclose(_lbfgsb, bugnicourt_mean, atol=1e-3)
    else:
        assert False
