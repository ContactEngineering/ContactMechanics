"""
Tests for tolerances, edge cases and hardness handling of the Polonsky & Keer
constrained conjugate gradient solver.
"""

import warnings

import numpy as np
import pytest
from SurfaceTopography import PlasticTopography, Topography, make_sphere

from ContactMechanics import FreeFFTElasticHalfSpace, PeriodicFFTElasticHalfSpace, make_system
from ContactMechanics.Optimization import ConvergenceWarning, constrained_conjugate_gradients
from ContactMechanics.PlasticSystemSpecialisations import PlasticNonSmoothContactSystem


def _rough_heights(nb_grid_pts, seed=1):
    """Self-affine random heights with unit rms height"""
    rng = np.random.default_rng(seed)
    q = np.fft.fftfreq(nb_grid_pts) * nb_grid_pts
    abs_q = np.sqrt(q.reshape(-1, 1) ** 2 + q.reshape(1, -1) ** 2)
    abs_q[0, 0] = 1
    amplitude = abs_q ** -1.8
    amplitude[0, 0] = 0
    amplitude[abs_q > nb_grid_pts / 4] = 0
    heights = np.fft.ifft2(amplitude * np.exp(2j * np.pi * rng.random(amplitude.shape))).real
    return heights / heights.std()


def _punch(nb_grid_pts=64, size=50., radius=20.):
    x = (np.arange(nb_grid_pts) - nb_grid_pts // 2) * size / nb_grid_pts
    r_sq = x.reshape(-1, 1) ** 2 + x.reshape(1, -1) ** 2
    return Topography(np.ma.masked_where(r_sq > radius ** 2, np.zeros((nb_grid_pts, nb_grid_pts))), (size, size))


def _sphere(nb_grid_pts=64):
    return make_sphere(1., (nb_grid_pts, nb_grid_pts), (1., 1.), centre=(0.5, 0.5), kind='paraboloid')


@pytest.mark.parametrize('young, length', [(1e9, 1e-6), (1e-9, 1e6)])
def test_unit_invariance(young, length):
    nb_grid_pts = 64
    heights = 0.01 * _rough_heights(nb_grid_pts)

    def solve(young, length):
        substrate = PeriodicFFTElasticHalfSpace((nb_grid_pts, nb_grid_pts), young, (length, length))
        return constrained_conjugate_gradients(substrate, Topography(heights * length, (length, length)),
                                               offset=0.)

    reference = solve(1., 1.)
    result = solve(young, length)
    assert result.success
    assert result.nit == reference.nit
    np.testing.assert_allclose(result.jac.sum() / (young * length ** 2), reference.jac.sum(), rtol=1e-10)
    np.testing.assert_array_equal(result.jac > 0, reference.jac > 0)


def test_flat_punch_at_constant_force_without_pentol():
    # The penetration tolerance must be inferred from the external force
    young, radius, force = 3.56, 20., 15.
    punch = _punch(radius=radius)
    result = constrained_conjugate_gradients(FreeFFTElasticHalfSpace(punch.nb_grid_pts, young, punch.physical_sizes),
                                             punch, external_force=force)
    assert result.success
    np.testing.assert_allclose(result.jac.sum(), force)
    # Contact stiffness of a flat punch is 2 a E*
    np.testing.assert_allclose(force / result.offset / (2 * radius * young), 1, rtol=0.05)


def test_flat_punch_not_in_contact():
    punch = _punch()
    result = constrained_conjugate_gradients(FreeFFTElasticHalfSpace(punch.nb_grid_pts, 1., punch.physical_sizes),
                                             punch, offset=-0.1)
    assert result.success
    np.testing.assert_array_equal(result.jac, 0)


def test_flat_topography_without_load_requires_pentol():
    punch = _punch()
    substrate = FreeFFTElasticHalfSpace(punch.nb_grid_pts, 1., punch.physical_sizes)
    with pytest.raises(ValueError):
        constrained_conjugate_gradients(substrate, punch, offset=0.)
    assert constrained_conjugate_gradients(substrate, punch, offset=0., pentol=1e-6).success


def test_zero_external_force():
    sphere = _sphere()
    result = constrained_conjugate_gradients(FreeFFTElasticHalfSpace(sphere.nb_grid_pts, 1., (1., 1.)), sphere,
                                             external_force=0., maxiter=100)
    assert result.success
    np.testing.assert_array_equal(result.jac, 0)


@pytest.mark.parametrize('substrate_class, offset', [(PeriodicFFTElasticHalfSpace, 0.),
                                                     (FreeFFTElasticHalfSpace, 0.002)])
def test_array_hardness(substrate_class, offset):
    if substrate_class is PeriodicFFTElasticHalfSpace:
        surface = Topography(0.01 * _rough_heights(64), (1., 1.))
    else:
        surface = _sphere()
    hardness = 0.02 * surface.area_per_pt

    def solve(hardness):
        substrate = substrate_class(surface.nb_grid_pts, 1., surface.physical_sizes)
        result = constrained_conjugate_gradients(substrate, surface, offset=offset, hardness=hardness)
        assert result.success
        return result

    # A uniform array must give the same result as a scalar
    scalar = solve(hardness)
    uniform = solve(np.full(surface.nb_grid_pts, hardness))
    np.testing.assert_array_equal(scalar.jac, uniform.jac)
    np.testing.assert_array_equal(scalar.plastic, uniform.plastic)
    assert scalar.plastic.sum() > 0

    # Spatially varying hardness is respected everywhere
    varying_hardness = np.full(surface.nb_grid_pts, hardness)
    varying_hardness[:surface.nb_grid_pts[0] // 2] /= 4
    varying = solve(varying_hardness)
    assert np.all(varying.jac <= varying_hardness * (1 + 1e-12))
    # Plastic points are at the hardness
    np.testing.assert_allclose(varying.jac[varying.plastic], varying_hardness[varying.plastic])


def test_elastic_contact_points_are_not_plastic():
    # Points in elastic contact can have a tiny negative gap within the
    # penetration tolerance; they must not be flagged as plastic.
    surface = _sphere()
    hardness = 0.02 * surface.area_per_pt
    result = constrained_conjugate_gradients(FreeFFTElasticHalfSpace(surface.nb_grid_pts, 1., (1., 1.)), surface,
                                             offset=0.002, hardness=hardness)
    assert result.success
    elastic_contact = np.logical_and(result.jac > 0, result.jac < hardness * (1 - 1e-12))
    assert elastic_contact.sum() > 0
    assert not np.any(result.plastic[elastic_contact])


def test_plastic_displacement_only_from_flowing_points():
    surface = _sphere()
    hardness = 0.02
    plastic_surface = PlasticTopography(surface, hardness)
    system = PlasticNonSmoothContactSystem(FreeFFTElasticHalfSpace(surface.nb_grid_pts, 1., (1., 1.)),
                                           plastic_surface)
    result = system.minimize_proxy(offset=0.002)
    assert result.success
    plastic_displ = plastic_surface.plastic_displ
    at_hardness = np.isclose(system.force, hardness * surface.area_per_pt)
    assert np.all(plastic_displ[np.logical_not(at_hardness)] == 0)
    assert np.any(plastic_displ[at_hardness] != 0)


def test_convergence_warning_and_state():
    surface = Topography(0.01 * _rough_heights(64), (1., 1.))
    system = make_system(substrate=PeriodicFFTElasticHalfSpace(surface.nb_grid_pts, 1., (1., 1.)), surface=surface)
    with pytest.warns(ConvergenceWarning):
        result = system.minimize_proxy(offset=0., maxiter=3)
    assert not result.success
    # The state is populated even without convergence
    assert system.disp is not None
    assert system.force is not None
    assert system.contact_zone is not None
    np.testing.assert_allclose(system.compute_normal_force(), result.jac.sum())


def test_check_boundaries():
    # A sphere pushed deep enough to touch the boundary of the domain
    surface = _sphere(32)
    offset = 0.5

    def solve(**kwargs):
        system = make_system(substrate=FreeFFTElasticHalfSpace(surface.nb_grid_pts, 1., (1., 1.), **kwargs),
                             surface=surface)
        return system.minimize_proxy(offset=offset)

    with pytest.warns(UserWarning, match='not zero at the boundary'):
        assert solve().success
    with pytest.raises(FreeFFTElasticHalfSpace.FreeBoundaryError):
        solve(check_boundaries=True)
    with warnings.catch_warnings():
        warnings.filterwarnings('error', message='.*not zero at the boundary')
        assert solve(check_boundaries=False).success


@pytest.mark.parametrize('seed', range(1, 7))
def test_convergence_when_rigid_surface_penetrates_everywhere(seed):
    # If the undeformed rigid surface penetrates the substrate everywhere, the
    # initial gap vanishes up to roundoff. This used to stall the iteration
    # (tiny but finite G_old, huge conjugacy factor G/G_old) with a wrong
    # contact area, depending on roundoff.
    heights = 0.01 * _rough_heights(64, seed=seed)
    surface = Topography(heights, (1., 1.))
    result = constrained_conjugate_gradients(PeriodicFFTElasticHalfSpace(surface.nb_grid_pts, 1., (1., 1.)),
                                             surface, offset=-heights.min(), maxiter=1000)
    assert result.success
    # Shifting the heights must not change the result
    shifted = constrained_conjugate_gradients(PeriodicFFTElasticHalfSpace(surface.nb_grid_pts, 1., (1., 1.)),
                                              Topography(heights + 5, (1., 1.)), offset=-heights.min() - 5,
                                              maxiter=1000)
    assert shifted.success
    np.testing.assert_allclose(shifted.jac.sum(), result.jac.sum(), rtol=1e-4)
