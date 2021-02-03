
# This aims to make a strong test for CG's using the simulation in figure 3 of
# the Bugnicourt paper but on reduced grid siye  and approx 20% contact area.
# Note that they use a nonadhesive simulation result to initialize the fields.

Es = 1.
rms_slope = 1.
dx = 1  # pixel size

from SurfaceTopography.Generation import fourier_synthesis
from Adhesion.Interactions import Exponential, HardWall
from Adhesion.System import make_system, BoundedSmoothContactSystem
import numpy as np
import matplotlib.pyplot as plt

# for the CM demo
from ContactMechanics.Tools.Logger import screen, Logger
import scipy.optimize as optim
import ContactMechanics as Solid
np.random.seed(0)
from NuMPI.Optimization import bugnicourt_cg


def test_bug_bench():
    # nondim we use
    Es = 1.
    rms_curvature = 1.
    w = 1.

    # this yields for example the systems of unit in bugnicourt equation 42

    # topography

    n = 2048

    # Rc ** 2/3 (w / Es)**(1/3)
    lateral_unit = (60e-9**(2/3) * (5e-3 / 25e6)**(1/3))

    s = 4e-6 / lateral_unit

    dx = s / n

    long_cutoff = s * 1 /4
    short_cutoff = s * 0.1 /4

    topo = fourier_synthesis(nb_grid_pts=(n, n),
                             hurst=0.8,
                             physical_sizes=(s, s),
                             rms_slope=1,
                             long_cutoff=long_cutoff,
                             short_cutoff=short_cutoff,
                             )

    topo = topo.scale(1 / topo.rms_curvature()).squeeze()


    # Adhesion parameters:

    tabor = 1.5
    interaction_length = 1 / tabor # original was 1/tabor.

    interaction = Exponential(w, interaction_length)
    process_zone_size = Es * w / np.pi / abs(interaction.max_tensile) ** 2

    substrate = Solid.PeriodicFFTElasticHalfSpace(nb_grid_pts=(n,n), young=Es, physical_sizes = (s, s), stiffness_q0 = 0.0)

    system = make_system(interaction=interaction,
                           surface=topo,
                           substrate='periodic',
                           young=Es,
                           system_class=BoundedSmoothContactSystem)

    # lazy system initialisation
    # system = make_system(interaction=interaction,
    #                      surface=topo,
    #                      substrate=substrate,
    #                      young=Es,
    #                      system_class=BoundedSmoothContactSystem)
    # substrate = system.substrate
    # system.substrate.stiffness_q0 = 0.0



    # TODO: actually use a Nonadhesive contact system
    nonadh_system = make_system(
         interaction=Exponential(0, interaction_length),
         surface=topo,
         substrate=substrate,
         system_class=BoundedSmoothContactSystem)

    # nonsmooth_system = Solid.Systems.NonSmoothContactSystem(substrate, topo)



    print("rms_slope: {}".format(topo.rms_slope()))
    print("nb pixels in process zone: {}".format(process_zone_size / dx))
    print("highcut wavelength / process_zone size: "
          "{}".format(short_cutoff / process_zone_size))
    print("rms_height / interaction_length: "
          "{}".format(topo.rms_height() / interaction_length))
    print("Persson and Tosatti adhesion criterion: ")
    print("   elastic deformation energy for full contact / work of adhesion: "
          "{}".format(substrate.evaluate(topo.detrend().heights())[0]
           / w / np.prod(topo.physical_sizes)))  # ^
    #            sets the mean to zero, so         |
    #            that the energy of the q = 0   --
    #            mode is not taken into
    #            account

    # P&R equation 10, with kappa_rep = 2, d_rep=4 h'rms / h''rms
    # (Only valid in the DMT regime, i.e. until the onset of stickiness)
    # If this value is smaller then 1 the surfaces are sticky and
    # normal CG's might not work anymore
    print("P&R stickiness criterion: "
    "{}".format(topo.rms_slope() * Es * interaction_length / 2 / w *
    (topo.rms_slope()**2/topo.rms_curvature() / interaction_length)**(2/3)))

    print("max height {}".format(np.max(system.surface.heights())))

    # %%
    # gradient tolerance with respect to a typical pixel force
    gtol = 1e-4

    disp0 = np.zeros(topo.nb_grid_pts)


    penetration = 30.0

    init_gap = disp0 - system.surface.heights() - penetration

    lbounds = np.zeros((n, n))
    bnds_gap = system._reshape_bounds(lbounds, )

    lbounds = system._lbounds_from_heights(penetration)
    bnds_disp = system._reshape_bounds(lbounds,)

    sol = optim.minimize(system.primal_objective(penetration, gradient=True),init_gap,
                         method = 'L-BFGS-B',jac=True, bounds = bnds_gap,
                         options=dict(gtol=gtol * max(Es * topo.rms_slope(),
        abs(interaction.max_tensile)) * topo.area_per_pt,ftol=0,maxcor=3),callback=None)

    print(sol.message, sol.nfev)
    assert sol.success, sol.message
    sol_lbfgs = sol
    mask = sol.x == 0
    print('lbfgsb frac ar. {}'.format(mask.sum() / (n * n)))
    sol_lbfgs_smooth = sol.x
    iter_lbfgsb = sol.nfev
    mean_val_lbfgs = np.mean(sol.x)


    res = bugnicourt_cg.constrained_conjugate_gradients(
        system.primal_objective(penetration, gradient=True),
        system.primal_hessian_product,
        x0=init_gap, mean_val=None,
        gtol=gtol * max(Es * topo.rms_slope(), abs(
                interaction.max_tensile)) * topo.area_per_pt,
        maxiter=1000)

    print(res.message, res.nit)
    assert res.success
    sol_bug = res.x

    print('min {} and max {} lbfgs'.format(np.min(sol_lbfgs_smooth), np.max(sol_lbfgs_smooth)))
    print('min {} and max {} bug'.format(np.min(sol_bug), np.max(sol_bug)))

    np.testing.assert_allclose(sol_lbfgs_smooth,sol_bug,atol=1e-2)
