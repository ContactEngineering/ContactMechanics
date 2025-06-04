import numpy as np
import scipy.optimize as optim

from NuMPI.Optimization import CCGWithoutRestart, CCGWithRestart
from SurfaceTopography import make_sphere

import ContactMechanics as Solid


def test_using_primal_obj():
    nx = ny = 128
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
    init_gap = np.zeros((nx, ny))
    disp = np.zeros((nx, ny))
    init_gap = disp - surface.heights() - offset

    # ####################POLONSKY-KEER##############################
    res = CCGWithRestart.constrained_conjugate_gradients(
        system.primal_objective(offset, gradient=True),
        system.primal_hessian_product, x0=init_gap, gtol=gtol)

    assert res.success
    polonsky_gap = res.x.reshape((nx, ny))

    # ####################BUGNICOURT###################################
    res = CCGWithoutRestart.constrained_conjugate_gradients(
        system.primal_objective(offset, gradient=True),
        system.primal_hessian_product, x0=init_gap, mean_val=None, gtol=gtol)
    assert res.success

    bugnicourt_gap = res.x.reshape((nx, ny))

    # #####################LBFGSB#####################################
    res = optim.minimize(system.primal_objective(offset, gradient=True),
                         system.shape_minimisation_input(init_gap),
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
    res = CCGWithRestart.constrained_conjugate_gradients(
        system.primal_objective(offset, gradient=True),
        system.primal_hessian_product, init_gap, gtol=gtol,
        mean_value=mean_val)

    assert res.success
    polonsky_gap_mean_cons = res.x.reshape((nx, ny))

    # ####################BUGNICOURT###################################
    CCGWithoutRestart.constrained_conjugate_gradients(system.primal_objective
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


def test_using_dual_obj():
    nx = ny = 128
    sx, sy = 1., 1.
    R = 10.

    gtol = 1e-7

    surface = make_sphere(R, (nx, ny), (sx, sy), kind="paraboloid")
    Es = 50.
    substrate = Solid.PeriodicFFTElasticHalfSpace((nx, ny), young=Es, physical_sizes=(sx, sy))

    system = Solid.Systems.NonSmoothContactSystem(substrate, surface)

    offset = 0.005
    lbounds = np.zeros((nx, ny))
    bnds = system._reshape_bounds(lbounds, )
    init_gap = np.zeros((nx, ny))
    disp = init_gap + surface.heights() + offset
    init_pressure = substrate.evaluate_force(disp)

    # ####################LBFGSB########################################
    res = optim.minimize(system.dual_objective(offset, gradient=True),
                         system.shape_minimisation_input(init_pressure),
                         method='L-BFGS-B', jac=True,
                         bounds=bnds,
                         options=dict(gtol=gtol, ftol=1e-20))

    assert res.success
    CA_lbfgsb = res.x.reshape((nx, ny)) > 0  # Contact area
    CA_lbfgsb = CA_lbfgsb.sum() / (nx * ny)
    lbfgsb_force = res.x.reshape((nx, ny))
    fun = system.dual_objective(offset, gradient=True)
    gap_lbfgsb = fun(res.x)[1]
    gap_lbfgsb = gap_lbfgsb.reshape((nx, ny))

    # ###################BUGNICOURT########################################
    CCGWithoutRestart.constrained_conjugate_gradients(
        system.dual_objective(offset, gradient=True),
        system.dual_hessian_product, init_pressure, mean_val=None, gtol=gtol)
    assert res.success

    bugnicourt_force = res.x.reshape((nx, ny))
    CA_bugnicourt = res.x.reshape((nx, ny)) > 0  # Contact area
    CA_bugnicourt = CA_bugnicourt.sum() / (nx * ny)
    gap_bugnicourt = fun(res.x)[1]
    gap_bugnicourt = gap_bugnicourt.reshape((nx, ny))
    #
    # # ##################POLONSKY-KEER#####################################
    res = CCGWithRestart.constrained_conjugate_gradients(
        system.dual_objective(offset, gradient=True),
        system.dual_hessian_product, init_pressure, gtol=gtol)
    assert res.success

    polonsky_force = res.x.reshape((nx, ny))
    CA_polonsky = res.x.reshape((nx, ny)) > 0  # Contact area
    CA_polonsky = CA_polonsky.sum() / (nx * ny)
    gap_polonsky = fun(res.x)[1]
    gap_polonsky = gap_polonsky.reshape((nx, ny))

    np.testing.assert_allclose(gap_lbfgsb, gap_polonsky, atol=1e-3)
    np.testing.assert_allclose(gap_lbfgsb, gap_bugnicourt, atol=1e-3)
    np.testing.assert_allclose(gap_bugnicourt, gap_polonsky, atol=1e-3)
    np.testing.assert_allclose(CA_lbfgsb, CA_polonsky, atol=1e-3)
    np.testing.assert_allclose(CA_lbfgsb, CA_bugnicourt, atol=1e-3)
    np.testing.assert_allclose(bugnicourt_force, polonsky_force, atol=1e-3)
    np.testing.assert_allclose(lbfgsb_force, polonsky_force, atol=1e-3)
    np.testing.assert_allclose(lbfgsb_force, bugnicourt_force, atol=1e-3)
    np.testing.assert_allclose(bugnicourt_force, polonsky_force, atol=1e-3)

    # ##########TEST MEAN VALUES#######################################
    mean_val = np.mean(lbfgsb_force)
    print('mean {}'.format(mean_val))
    # ####################POLONSKY-KEER##############################
    res = CCGWithRestart.constrained_conjugate_gradients(
        system.dual_objective(offset, gradient=True),
        system.dual_hessian_product, init_pressure, gtol=gtol,
        mean_value=mean_val)

    assert res.success
    polonsky_mean = res.x.reshape((nx, ny))

    # # ####################BUGNICOURT###################################
    CCGWithoutRestart.constrained_conjugate_gradients(
        system.dual_objective(offset, gradient=True),
        system.dual_hessian_product, init_pressure, mean_val=mean_val,
        gtol=gtol)
    assert res.success

    bugnicourt_mean = res.x.reshape((nx, ny))

    np.testing.assert_allclose(polonsky_mean, lbfgsb_force, atol=1e-3)
    np.testing.assert_allclose(bugnicourt_mean, lbfgsb_force, atol=1e-3)


def test_dual_obj_nonperiodic():
    nx = ny = 128
    sx, sy = 1., 1.
    R = 10.

    gtol = 1e-7

    surface = make_sphere(R, (nx, ny), (sx, sy), kind="paraboloid")
    Es = 50.
    substrate = Solid.FreeFFTElasticHalfSpace((nx, ny), young=Es, physical_sizes=(sx, sy))

    system = Solid.Systems.NonSmoothContactSystem(substrate, surface)

    offset = 0.005
    # lbounds = np.zeros((2 * nx, 2 * ny))
    #
    # Inpose contact constraint only where we have topopgraphy, i.e. not in the padding region
    # lbounds = np.ma.masked_all(system.substrate.nb_subdomain_grid_pts)
    # lbounds.mask[system.substrate.local_topography_subdomain_slices] = False
    # lbounds[system.substrate.local_topography_subdomain_slices] = 0
    # lbounds.set_fill_value(-np.inf)

    ######

    bnds = optim.Bounds(lb=np.zeros(np.prod(system.substrate.topography_nb_subdomain_grid_pts)))
    # system._reshape_bounds(lbounds, )
    # init_gap = np.zeros((nx, ny))
    # disp = np.zeros((2 * nx, 2 * ny))
    init_pressure = np.zeros((nx,  ny))

    # ####################LBFGSB########################################
    res = optim.minimize(system.dual_objective(offset, gradient=True),
                         init_pressure.reshape(-1),
                         method='L-BFGS-B', jac=True,
                         bounds=bnds,
                         options=dict(gtol=gtol, ftol=1e-20))

    assert res.success

    # TODO: correct computation of the contact area etc...
    CA_lbfgsb = res.x.reshape((nx, ny)) > 0  # Contact area
    CA_lbfgsb = CA_lbfgsb.sum() / (nx * ny)
    lbfgsb_force = res.x.reshape((nx, ny))
    fun = system.dual_objective(offset, gradient=True)
    gap_lbfgsb = fun(res.x)[1]
    gap_lbfgsb = gap_lbfgsb.reshape((nx, ny))

    # ###################BUGNICOURT########################################
    CCGWithoutRestart.constrained_conjugate_gradients(
        system.dual_objective(offset, gradient=True),
        system.dual_hessian_product, init_pressure, mean_val=None, gtol=gtol)
    assert res.success

    bugnicourt_force = res.x.reshape((nx, ny))
    CA_bugnicourt = res.x.reshape((nx, ny)) > 0  # Contact area
    CA_bugnicourt = CA_bugnicourt.sum() / (nx * ny)
    gap_bugnicourt = fun(res.x)[1]
    gap_bugnicourt = gap_bugnicourt.reshape((nx, ny))
    #
    # ##################POLONSKY-KEER#####################################
    #
    # class WRAPOBJ():
    #     def __init__(self, ):
    #         self.original_fun = system.dual_objective(offset, gradient=True)
    #         fig, [self.ax, self.axpres] = plt.subplots(2, 1)
    #         # plt.show()
    #         self.ax.set_yscale('log')
    #         self.it = 0
    #
    #     def __call__(self, x):
    #         en, grad = self.original_fun(x)
    #         rms_grad = np.sqrt(system.reduction.sum((grad * (x > 0)) ** 2))
    #         print(rms_grad)
    #         self.ax.plot(self.it, rms_grad, ".")
    #         self.it += 1
    #         self.ax.relim()
    #         self.axpres.clear()
    #         self.axpres.imshow(x.reshape((nx, ny)))
    #         plt.draw()
    #         plt.pause(0.01)
    #         return en, grad

    res = CCGWithRestart.constrained_conjugate_gradients(
        system.dual_objective(offset, gradient=True),
        # WRAPOBJ(),
        system.dual_hessian_product,
        init_pressure + 0.01,  # This algorithm gets stuck if the initial pressure is perfectly 0
        gtol=gtol)
    assert res.success

    polonsky_force = res.x.reshape((nx, ny))
    CA_polonsky = res.x.reshape((nx, ny)) > 0  # Contact area
    CA_polonsky = CA_polonsky.sum() / (nx * ny)
    gap_polonsky = fun(res.x)[1]
    gap_polonsky = gap_polonsky.reshape((nx, ny))

    np.testing.assert_allclose(gap_lbfgsb, gap_polonsky, atol=1e-3)
    np.testing.assert_allclose(gap_lbfgsb, gap_bugnicourt, atol=1e-3)
    np.testing.assert_allclose(gap_bugnicourt, gap_polonsky, atol=1e-3)
    np.testing.assert_allclose(CA_lbfgsb, CA_polonsky, atol=1e-3)
    np.testing.assert_allclose(CA_lbfgsb, CA_bugnicourt, atol=1e-3)
    np.testing.assert_allclose(bugnicourt_force, polonsky_force, atol=1e-3)
    np.testing.assert_allclose(lbfgsb_force, polonsky_force, atol=1e-3)
    np.testing.assert_allclose(lbfgsb_force, bugnicourt_force, atol=1e-3)
    np.testing.assert_allclose(bugnicourt_force, polonsky_force, atol=1e-3)

    # ##########TEST MEAN VALUES#######################################
    mean_val = np.mean(lbfgsb_force)
    print('mean {}'.format(mean_val))
    # ####################POLONSKY-KEER##############################
    res = CCGWithRestart.constrained_conjugate_gradients(
        system.dual_objective(offset, gradient=True),
        system.dual_hessian_product, init_pressure + 0.01, gtol=gtol,
        mean_value=mean_val)

    assert res.success
    polonsky_mean = res.x.reshape((nx, ny))

    # # ####################BUGNICOURT###################################
    CCGWithoutRestart.constrained_conjugate_gradients(
        system.dual_objective(offset, gradient=True),
        system.dual_hessian_product, init_pressure, mean_val=mean_val,
        gtol=gtol)
    assert res.success

    bugnicourt_mean = res.x.reshape((nx, ny))

    np.testing.assert_allclose(polonsky_mean, lbfgsb_force, atol=1e-3)
    np.testing.assert_allclose(bugnicourt_mean, lbfgsb_force, atol=1e-3)
#
#
# def test_primal_obj_nonperiodic():
#     # Note that there is a bit of overlap with test_ccg_without_restart_nonperiodic
#
#     nx = ny = 128
#     sx, sy = 1., 1.
#     R = 10.
#
#     gtol = 1e-6
#
#     surface = make_sphere(R, (nx, ny), (sx, sy), kind="paraboloid")
#     Es = 50.
#     substrate = Solid.FreeFFTElasticHalfSpace((nx, ny), young=Es,
#                                                   physical_sizes=(sx, sy))
#
#     system = Solid.Systems.NonSmoothContactSystem(substrate, surface)
#
#     offset = 0.005
#
#     lbounds = np.ma.masked_all(self.substrate.nb_subdomain_grid_pts)
#     lbounds.mask[self.substrate.local_topography_subdomain_slices] = False
#     lbounds[self.substrate.local_topography_subdomain_slices] = self.surface.heights() + offset
#
#     lbounds.set_fill_value(-np.inf)
#
#     init_gap =
#     disp = np.zeros((nx, ny))
#     init_gap[substrate.topography_subdomain_slices] = disp - surface.heights() - offset
#
#
#
#
#     # ####################POLONSKY-KEER##############################
#     res = CCGWithRestart.constrained_conjugate_gradients(
#         system.primal_objective(offset, gradient=True),
#         system.primal_hessian_product, x0=init_gap, gtol=gtol)
#
#     assert res.success
#     polonsky_gap = res.x.reshape((nx, ny))
#
#     # ####################BUGNICOURT###################################
#     res = CCGWithoutRestart.constrained_conjugate_gradients(
#         system.primal_objective(offset, gradient=True),
#         system.primal_hessian_product, x0=init_gap, mean_val=None, gtol=gtol)
#     assert res.success
#
#     bugnicourt_gap = res.x.reshape((nx, ny))
#
#     # #####################LBFGSB#####################################
#     res = optim.minimize(system.primal_objective(offset, gradient=True),
#                          system.shape_minimisation_input(init_gap),
#                          method='L-BFGS-B', jac=True,
#                          bounds=bnds,
#                          options=dict(gtol=gtol, ftol=1e-20))
#
#     assert res.success
#     lbfgsb_gap = res.x.reshape((nx, ny))
#
#     np.testing.assert_allclose(polonsky_gap, bugnicourt_gap, atol=1e-3)
#     np.testing.assert_allclose(polonsky_gap, lbfgsb_gap, atol=1e-3)
#     np.testing.assert_allclose(lbfgsb_gap, bugnicourt_gap, atol=1e-3)
#
#     # ##########TEST MEAN VALUES#######################################
#     mean_val = np.mean(lbfgsb_gap)
#     # ####################POLONSKY-KEER##############################
#     res = CCGWithRestart.constrained_conjugate_gradients(
#         system.primal_objective(offset, gradient=True),
#         system.primal_hessian_product, init_gap, gtol=gtol,
#         mean_value=mean_val)
#
#     assert res.success
#     polonsky_gap_mean_cons = res.x.reshape((nx, ny))
#
#     # ####################BUGNICOURT###################################
#     CCGWithoutRestart.constrained_conjugate_gradients(system.primal_objective
#                                                       (offset, gradient=True),
#                                                       system.
#                                                       primal_hessian_product,
#                                                       x0=init_gap,
#                                                       mean_val=mean_val,
#                                                       gtol=gtol
#                                                       )
#     assert res.success
#
#     bugnicourt_gap_mean_cons = res.x.reshape((nx, ny))
#
#     np.testing.assert_allclose(polonsky_gap_mean_cons, lbfgsb_gap, atol=1e-3)
#     np.testing.assert_allclose(bugnicourt_gap_mean_cons, lbfgsb_gap, atol=1e-3)
#     np.testing.assert_allclose(lbfgsb_gap, bugnicourt_gap, atol=1e-3)
#     np.testing.assert_allclose(lbfgsb_gap, bugnicourt_gap_mean_cons, atol=1e-3)
