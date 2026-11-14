"""
Tests of the sign conventions of forces in contact systems (#28): the
substrate force is -dE/du on the grid of the substrate, independently of how
the state was computed, and the normal load is positive for compressive
contact.
"""

import numpy as np
import pytest
from SurfaceTopography import make_sphere

from ContactMechanics import FreeFFTElasticHalfSpace, PeriodicFFTElasticHalfSpace
from ContactMechanics.Systems import NonSmoothContactSystem


def _system(substrate_class):
    surface = make_sphere(10., (32, 32), (1., 1.), centre=(0.5, 0.5), kind='paraboloid')
    return NonSmoothContactSystem(substrate_class((32, 32), 50., (1., 1.)), surface)


def _solve(system, method):
    offset = 0.002
    if method == 'polonsky-keer':
        result = system.minimize_proxy(offset=offset, pentol=1e-10)
    elif method == 'primal':
        result = system.primal_minimize_proxy(offset)
    else:
        result = system.dual_minimize_proxy(offset)
    assert result.success
    return result


@pytest.mark.parametrize('substrate_class', [PeriodicFFTElasticHalfSpace, FreeFFTElasticHalfSpace])
@pytest.mark.parametrize('method', ['polonsky-keer', 'primal', 'dual'])
def test_state_after_minimization(substrate_class, method):
    system = _system(substrate_class)
    _solve(system, method)
    load = system.compute_normal_force()
    assert load > 0
    # Contact force on the topography grid, positive in compression
    assert system.force.shape == system.surface.nb_grid_pts
    np.testing.assert_allclose(system.force.sum(), load)
    # (The primal solver yields forces outside the contact up to its tolerance)
    assert system.force.min() >= -1e-4 * system.force.max()
    # Substrate force on the grid of the substrate, opposite sign
    assert system.substrate.force.shape == system.substrate.nb_subdomain_grid_pts
    np.testing.assert_allclose(system.substrate.force[system.comp_slice], -system.force)
    np.testing.assert_allclose(system.energy, system.substrate.energy)


@pytest.mark.parametrize('substrate_class', [PeriodicFFTElasticHalfSpace, FreeFFTElasticHalfSpace])
def test_normal_force_independent_of_path(substrate_class):
    reference = _system(substrate_class)
    _solve(reference, 'polonsky-keer')
    load = reference.compute_normal_force()

    # Evaluating the elastic response of the solution gives the same load
    system = _system(substrate_class)
    system.evaluate(reference.disp, reference.offset, forces=True)
    np.testing.assert_allclose(system.compute_normal_force(), load, rtol=1e-6)

    # So do all other solvers
    for method in ['primal', 'dual']:
        system = _system(substrate_class)
        _solve(system, method)
        np.testing.assert_allclose(system.compute_normal_force(), load, rtol=1e-4)


def test_dual_objective_substrate_force():
    system = _system(FreeFFTElasticHalfSpace)
    force = np.zeros(system.surface.nb_grid_pts)
    force[15:17, 15:17] = 1.
    system.dual_objective(0.)(force.ravel())
    assert system.substrate.force.shape == system.substrate.nb_subdomain_grid_pts
    np.testing.assert_allclose(system.compute_normal_force(), force.sum())


def test_minimize_proxy_result():
    # Attributes documented in minimize_proxy
    system = _system(FreeFFTElasticHalfSpace)
    result = _solve(system, 'polonsky-keer')
    assert result.x.shape == system.substrate.nb_subdomain_grid_pts
    assert result.jac.shape == system.surface.nb_grid_pts
    np.testing.assert_allclose(result.fun, (result.jac * result.x[system.comp_slice]).sum() / 2)
    for attribute in ['success', 'message', 'offset', 'active_set', 'nit', 'nfev', 'maxcv']:
        assert hasattr(result, attribute)
