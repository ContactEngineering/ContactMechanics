"""
Tests of the generic interface of contact systems.
"""

import numpy as np
import pytest
import scipy.optimize as optim
from SurfaceTopography import PlasticTopography, Topography, make_sphere

from ContactMechanics import FreeFFTElasticHalfSpace, PeriodicFFTElasticHalfSpace
from ContactMechanics.PlasticSystemSpecialisations import PlasticNonSmoothContactSystem
from ContactMechanics.Systems import IncompatibleResolutionError, NonSmoothContactSystem, SystemBase


def _sphere_system(substrate_class=FreeFFTElasticHalfSpace, nb_grid_pts=32):
    surface = make_sphere(1., (nb_grid_pts, nb_grid_pts), (1., 1.), centre=(0.5, 0.5), kind='paraboloid')
    return NonSmoothContactSystem(substrate_class((nb_grid_pts, nb_grid_pts), 1., (1., 1.)), surface)


@pytest.mark.parametrize('substrate_class', [PeriodicFFTElasticHalfSpace, FreeFFTElasticHalfSpace])
def test_scipy_minimize_proxy(substrate_class):
    # The generic scipy-based minimization of the elastic energy with the
    # constraint as bounds reproduces the Polonsky & Keer solution
    offset = 0.002
    reference = _sphere_system(substrate_class)
    assert reference.minimize_proxy(offset=offset, pentol=1e-12).success

    system = _sphere_system(substrate_class)
    result = SystemBase.minimize_proxy(system, offset=offset, lbounds='auto', method='L-BFGS-B',
                                       options=dict(gtol=1e-12, ftol=0, maxiter=10000))
    assert result.success
    assert result.x.shape == system.substrate.nb_subdomain_grid_pts
    np.testing.assert_allclose(system.disp, reference.disp, atol=1e-6)
    np.testing.assert_allclose(system.compute_normal_force(), reference.compute_normal_force(), rtol=1e-3)


def test_scipy_minimize_proxy_options():
    system = _sphere_system()
    with pytest.raises(ValueError):
        SystemBase.minimize_proxy(system, offset=0.002, lbounds='invalid')
    with pytest.raises(NotImplementedError):
        SystemBase.minimize_proxy(system, offset=0.002, lbounds='auto', callback=True)
    # Without bounds, the elastic energy is minimal for zero displacement
    result = SystemBase.minimize_proxy(system, offset=0.002, method='CG')
    assert result.success
    np.testing.assert_allclose(system.disp, 0, atol=1e-8)


def test_bounds():
    system = _sphere_system(PeriodicFFTElasticHalfSpace, 4)
    lower = np.zeros((4, 4))
    upper = np.ones((4, 4))
    bounds = system._reshape_bounds(lower, upper)
    np.testing.assert_array_equal(bounds.lb, 0)
    np.testing.assert_array_equal(bounds.ub, 1)
    assert isinstance(system._reshape_bounds(ubounds=upper), optim.Bounds)
    assert isinstance(system._reshape_bounds(lbounds=lower), optim.Bounds)
    assert system._reshape_bounds() is None


def test_shapes():
    system = _sphere_system(PeriodicFFTElasticHalfSpace, 4)
    np.testing.assert_array_equal(system.shape_minimisation_output(np.arange(16.)), np.arange(16.).reshape(4, 4))
    with pytest.raises(IncompatibleResolutionError):
        system.shape_minimisation_input(np.zeros(5))
    with pytest.raises(IncompatibleResolutionError):
        system.shape_minimisation_output(np.zeros(5))
    with pytest.raises(IncompatibleResolutionError):
        NonSmoothContactSystem(PeriodicFFTElasticHalfSpace((4, 4), 1., (1., 1.)),
                               Topography(np.zeros((5, 4)), (1., 1.)))


def test_contact_area():
    system = _sphere_system()
    assert system.minimize_proxy(offset=0.002).success
    nb_contact_pts = system.compute_nb_contact_pts()
    assert nb_contact_pts > 0
    np.testing.assert_allclose(system.compute_contact_area(), nb_contact_pts * system.area_per_pt)
    np.testing.assert_allclose(system.compute_relative_contact_area(), nb_contact_pts / 32 ** 2)
    assert len(system.compute_contact_coordinates()) == nb_contact_pts
    assert not system.is_proxy()
    assert system.nb_grid_pts == (32, 32)


def test_handles():
    surface = Topography(np.zeros((4, 4)), (1., 1.))
    assert NonSmoothContactSystem.handles(PeriodicFFTElasticHalfSpace, type(surface), False)
    assert not NonSmoothContactSystem.handles(int, type(surface), False)
    plastic_surface = PlasticTopography(surface, 1.)
    assert PlasticNonSmoothContactSystem.handles(PeriodicFFTElasticHalfSpace, type(plastic_surface), False)
    assert not PlasticNonSmoothContactSystem.handles(PeriodicFFTElasticHalfSpace, type(surface), False)
    assert not SystemBase.handles(PeriodicFFTElasticHalfSpace, type(surface), False)


def test_objectives():
    system = _sphere_system(PeriodicFFTElasticHalfSpace, 16)
    disp = np.random.default_rng(0).random(16 * 16) * 1e-3
    energy, gradient = system.objective(0.002, gradient=True)(disp)
    np.testing.assert_allclose(system.objective(0.002, gradient=False)(disp), energy)
    energy, gradient = system.primal_objective(0.002, gradient=True)(disp)
    np.testing.assert_allclose(system.primal_objective(0.002, gradient=False)(disp), energy)
    np.testing.assert_allclose(system.hessian_product(disp), system.primal_hessian_product(disp))
    # Wrong shapes give an informative error
    for objective in [system.objective(0.002, gradient=True), system.primal_objective(0.002, gradient=True),
                      system.dual_objective(0.002, gradient=True)]:
        with pytest.raises(ValueError, match='res'):
            objective(np.zeros(7))
    with pytest.raises(NotImplementedError):
        system.dual_objective(0.002, gradient=False)
    # Without forces, the force is not stored
    system.evaluate(disp.reshape(16, 16), 0.002, forces=False)
    assert system.force is None


def test_proxy_arguments():
    system = _sphere_system(PeriodicFFTElasticHalfSpace, 16)
    with pytest.raises(ValueError):
        system.primal_minimize_proxy(0.002, solver='unknown')
    with pytest.raises(ValueError):
        system.dual_minimize_proxy(0.002, solver='unknown')
    with pytest.raises(IncompatibleResolutionError):
        system.dual_minimize_proxy(0.002, init_force=np.zeros((5, 5)))


def test_length_scale_of_flat_surface():
    system = NonSmoothContactSystem(FreeFFTElasticHalfSpace((16, 16), 1., (1., 1.)),
                                    Topography(np.zeros((16, 16)), (1., 1.)))
    np.testing.assert_allclose(system._length_scale(0.1), 0.1)
    with pytest.raises(ValueError):
        system._length_scale(0.)


def test_abstract_interface():
    class Minimal(SystemBase):
        def evaluate(self, disp, offset, pot=True, forces=False):
            return super().evaluate(disp, offset, pot, forces)

        def compute_normal_force(self):
            return super().compute_normal_force()

        def compute_nb_contact_pts(self):
            return super().compute_nb_contact_pts()

        def objective(self, offset, disp0=None, gradient=False, logger=None):
            return super().objective(offset, disp0, gradient, logger)

    system = Minimal(PeriodicFFTElasticHalfSpace((4, 4), 1., (1., 1.)), Topography(np.zeros((4, 4)), (1., 1.)))
    assert system.nb_grid_pts is None
    for method, args in [(system.evaluate, (None, 0)), (system.compute_normal_force, ()),
                         (system.compute_nb_contact_pts, ()), (system.objective, (0,)), (system.callback, ())]:
        with pytest.raises(NotImplementedError):
            method(*args)
