"""
Tests for the primal and dual minimize proxies of NonSmoothContactSystem.
After minimization, the state of the system (forces, displacements, gap,
contact area, energy) must agree with the Polonsky & Keer solution,
independently of the solver, the system of units, the boundary conditions
and the MPI decomposition.
"""

import numpy as np
import pytest
from NuMPI import MPI
from NuMPI.Tools import Reduction
from SurfaceTopography import Topography, UniformLineScan, make_sphere

from ContactMechanics import FreeFFTElasticHalfSpace, PeriodicFFTElasticHalfSpace
from ContactMechanics.Systems import NonSmoothContactSystem
from ContactMechanics.Tools.Logger import Logger

SOLVERS = ['ccg-without-restart', 'ccg-with-restart', 'l-bfgs-b']


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


def _periodic_system(young=1., length=1., nb_grid_pts=32):
    heights = 0.01 * length * _rough_heights(nb_grid_pts)
    substrate = PeriodicFFTElasticHalfSpace((nb_grid_pts, nb_grid_pts), young, (length, length))
    return NonSmoothContactSystem(substrate, Topography(heights, (length, length), periodic=True))


def _free_system(nb_grid_pts=32):
    sx = 1.
    surface = make_sphere(10., (nb_grid_pts, nb_grid_pts), (sx, sx), centre=(sx / 2, sx / 2), kind="paraboloid")
    substrate = FreeFFTElasticHalfSpace((nb_grid_pts, nb_grid_pts), 50., (sx, sx))
    return NonSmoothContactSystem(substrate, surface)


@pytest.mark.parametrize('kind', ['primal', 'dual'])
@pytest.mark.parametrize('solver', SOLVERS)
@pytest.mark.parametrize('make_system, offset', [(_periodic_system, 0.), (_free_system, 0.002)])
def test_state_agrees_with_polonsky_keer(kind, solver, make_system, offset):
    reference = make_system()
    assert reference.minimize_proxy(offset=offset, pentol=1e-10).success
    length = reference.surface.rms_height_from_area()

    system = make_system()
    result = getattr(system, f'{kind}_minimize_proxy')(offset, solver=solver)
    assert result.success, result.message

    np.testing.assert_allclose(system.force.sum(), reference.force.sum(), rtol=1e-4)
    assert system.disp.shape == reference.disp.shape
    np.testing.assert_allclose(system.disp, reference.disp, atol=1e-4 * length)
    np.testing.assert_allclose(system.energy, (reference.force * reference.disp[system.comp_slice]).sum() / 2,
                               rtol=1e-4)
    assert np.sum(system.contact_zone != reference.contact_zone) <= 0.01 * np.sum(reference.contact_zone)

    # Complementarity: no penetration, no force outside of the contact
    assert system.gap.min() > -1e-4 * length
    np.testing.assert_allclose(system.gap[system.contact_zone], 0, atol=1e-4 * length)
    assert np.abs(system.force[np.logical_not(system.contact_zone)]).max() <= \
        1e-4 * np.abs(system.force).max()


@pytest.mark.parametrize('kind', ['primal', 'dual'])
@pytest.mark.parametrize('solver', SOLVERS)
@pytest.mark.parametrize('young, length', [(1e9, 1e-6), (1e-9, 1e6)])
def test_unit_invariance(kind, solver, young, length):
    reference = _periodic_system()
    reference_result = getattr(reference, f'{kind}_minimize_proxy')(0., solver=solver)
    assert reference_result.success

    system = _periodic_system(young, length)
    result = getattr(system, f'{kind}_minimize_proxy')(0., solver=solver)
    assert result.success, result.message

    if solver != 'l-bfgs-b':
        # The conjugate-gradient iterations are scale invariant
        assert result.nit == reference_result.nit
    np.testing.assert_allclose(system.force.sum() / (young * length ** 2), reference.force.sum(), rtol=1e-5)
    np.testing.assert_allclose(system.disp / length, reference.disp, atol=1e-6)
    np.testing.assert_array_equal(system.contact_zone, reference.contact_zone)


@pytest.mark.parametrize('kind', ['primal', 'dual'])
def test_logger(kind, tmp_path):
    fn = tmp_path / 'log.out'
    system = _periodic_system()
    result = getattr(system, f'{kind}_minimize_proxy')(0., logger=Logger(str(fn)))
    assert result.success
    with open(fn) as f:
        assert len(f.readlines()) > result.nit


def test_primal_energy_only_objective_with_logger(tmp_path):
    fn = tmp_path / 'log.out'
    system = _periodic_system()
    res = system.substrate.nb_subdomain_grid_pts
    fun = system.primal_objective(0., gradient=False, logger=Logger(str(fn)))
    energy = fun(np.zeros(res).reshape(-1))
    np.testing.assert_allclose(energy, system.primal_objective(0.)(np.zeros(res).reshape(-1))[0])
    with open(fn) as f:
        assert len(f.readlines()) > 0


def test_primal_initial_gap_on_topography_domain():
    system = _free_system()
    nb_grid_pts = system.substrate.topography_nb_subdomain_grid_pts
    result = system.primal_minimize_proxy(0.002, init_gap=np.zeros(nb_grid_pts))
    assert result.success
    assert system.disp.shape == system.substrate.nb_subdomain_grid_pts


@pytest.mark.parametrize('kind', ['primal', 'dual'])
@pytest.mark.parametrize('solver', ['ccg-without-restart', 'ccg-with-restart'])
@pytest.mark.parametrize('substrate_class', [PeriodicFFTElasticHalfSpace, FreeFFTElasticHalfSpace])
def test_parallel(comm, kind, solver, substrate_class):
    nb_grid_pts = (32, 32)
    physical_sizes = (1., 1.)
    centre = (0.5, 0.5)
    offset = 0.002

    # Serial reference
    surface = make_sphere(10., nb_grid_pts, physical_sizes, centre=centre, kind="paraboloid",
                          communicator=MPI.COMM_SELF)
    substrate = substrate_class(nb_grid_pts, 50., physical_sizes, communicator=MPI.COMM_SELF)
    reference = NonSmoothContactSystem(substrate, surface)
    assert getattr(reference, f'{kind}_minimize_proxy')(offset, solver=solver).success

    # Domain decomposed calculation
    substrate = substrate_class(nb_grid_pts, 50., physical_sizes, communicator=comm, fft='mpi')
    surface = make_sphere(10., nb_grid_pts, physical_sizes, centre=centre, kind="paraboloid",
                          subdomain_locations=substrate.topography_subdomain_locations,
                          nb_subdomain_grid_pts=substrate.topography_nb_subdomain_grid_pts,
                          communicator=comm)
    system = NonSmoothContactSystem(substrate, surface)
    result = getattr(system, f'{kind}_minimize_proxy')(offset, solver=solver)
    assert result.success, result.message

    pnp = Reduction(comm)
    length = reference.surface.rms_height_from_area()
    np.testing.assert_allclose(pnp.sum(system.force), reference.force.sum(), rtol=1e-6)
    np.testing.assert_allclose(system.energy, reference.energy, rtol=1e-6)
    assert pnp.max(np.abs(system.disp - reference.disp[substrate.subdomain_slices])) < 1e-6 * length
    assert pnp.max(np.abs(system.force - reference.force[substrate.topography_subdomain_slices])) < \
        1e-6 * np.abs(reference.force).max()


@pytest.mark.parametrize('kind', ['primal', 'dual'])
def test_line_scan(kind):
    # Default tolerances for one-dimensional problems (line scans have no
    # areal rms height)
    nb_grid_pts = 256
    x = np.arange(nb_grid_pts) / nb_grid_pts
    surface = UniformLineScan(0.01 * np.cos(2 * np.pi * x), 1., periodic=True)
    system = NonSmoothContactSystem(PeriodicFFTElasticHalfSpace((nb_grid_pts,), 1., (1.,)), surface)
    reference = NonSmoothContactSystem(PeriodicFFTElasticHalfSpace((nb_grid_pts,), 1., (1.,)), surface)
    assert reference.minimize_proxy(offset=0., pentol=1e-10).success
    result = getattr(system, f'{kind}_minimize_proxy')(0.)
    assert result.success
    np.testing.assert_allclose(system.force.sum(), reference.force.sum(), rtol=1e-4)


@pytest.mark.parametrize('kind', ['primal', 'dual'])
@pytest.mark.parametrize('solver', ['ccg-without-restart', 'ccg-with-restart'])
def test_line_scan_unit_invariance(kind, solver):
    nb_grid_pts = 256
    x = np.arange(nb_grid_pts) / nb_grid_pts

    def solve(young, length):
        surface = UniformLineScan(0.01 * length * np.cos(2 * np.pi * x), length, periodic=True)
        system = NonSmoothContactSystem(PeriodicFFTElasticHalfSpace((nb_grid_pts,), young, (length,)), surface)
        result = getattr(system, f'{kind}_minimize_proxy')(0., solver=solver)
        assert result.success
        return result, system

    reference_result, reference = solve(1., 1.)
    for young, length in [(1e9, 1e-6), (1e-9, 1e6)]:
        result, system = solve(young, length)
        assert result.nit == reference_result.nit
        # Forces of line scans are per unit length, the load scales as E* L
        np.testing.assert_allclose(system.force.sum() / (young * length), reference.force.sum(), rtol=1e-6)


def _masked_punch(fill, height=0.):
    nb_grid_pts, size, radius = 32, 50., 20.
    x = (np.arange(nb_grid_pts) - nb_grid_pts // 2) * size / nb_grid_pts
    r_sq = x.reshape(-1, 1) ** 2 + x.reshape(1, -1) ** 2
    heights = np.full((nb_grid_pts, nb_grid_pts), fill)
    heights[r_sq <= radius ** 2] = height
    surface = Topography(np.ma.masked_where(r_sq > radius ** 2, heights), (size, size))
    return NonSmoothContactSystem(FreeFFTElasticHalfSpace(surface.nb_grid_pts, 1., (size, size)), surface)


@pytest.mark.parametrize('kind', ['primal', 'dual'])
@pytest.mark.parametrize('fill', [5., np.nan])
def test_masked_topography(kind, fill):
    # Values under the mask must not matter: the heights are undefined there
    reference = _masked_punch(0.)
    assert reference.minimize_proxy(offset=0.1, pentol=1e-10).success
    system = _masked_punch(fill)
    result = getattr(system, f'{kind}_minimize_proxy')(0.1)
    assert result.success
    np.testing.assert_allclose(system.force.sum(), reference.force.sum(), rtol=1e-5)
    np.testing.assert_array_equal(system.contact_zone, reference.contact_zone)


def test_primal_masked_initial_gap():
    # Values hidden under the mask of an initial guess must not matter
    reference = _masked_punch(np.nan)
    assert reference.primal_minimize_proxy(0.1).success
    system = _masked_punch(np.nan)
    init_gap = np.ma.masked_array(np.full(system.surface.nb_subdomain_grid_pts, np.nan),
                                  mask=np.ma.getmaskarray(system.surface.heights()))
    init_gap[~init_gap.mask] = 0.
    result = system.primal_minimize_proxy(0.1, init_gap=init_gap)
    assert result.success
    assert np.all(np.isfinite(system.force))
    np.testing.assert_allclose(system.force.sum(), reference.force.sum(), rtol=1e-5)


@pytest.mark.parametrize('kind', ['primal', 'dual'])
def test_flat_surface_with_inexact_height(kind):
    # The mean of a height that is not exactly representable is rounded; the
    # surface is nevertheless flat, so the tolerance follows from the
    # indentation depth
    system = _masked_punch(0., height=1.1)
    np.testing.assert_allclose(system._length_scale(-1.05), 0.05, rtol=1e-12)
    assert getattr(system, f'{kind}_minimize_proxy')(-1.05).success


@pytest.mark.parametrize('kind', ['primal', 'dual'])
def test_flat_surface_shift_invariance(kind):
    # A flat surface at height 0.5 and offset -0.4 is the same problem as at
    # height 0 and offset 0.1, including the default tolerance
    reference = _masked_punch(0.)
    assert getattr(reference, f'{kind}_minimize_proxy')(0.1).success
    system = _masked_punch(0., height=0.5)
    np.testing.assert_allclose(system._length_scale(-0.4), reference._length_scale(0.1), rtol=1e-12)
    result = getattr(system, f'{kind}_minimize_proxy')(-0.4)
    assert result.success
    np.testing.assert_allclose(system.force.sum(), reference.force.sum(), rtol=1e-6)


@pytest.mark.parametrize('kind', ['primal', 'dual'])
def test_substrate_with_young_only(kind):
    # Custom substrates may define Young's modulus only
    class Substrate(PeriodicFFTElasticHalfSpace):
        hidden = False

        @property
        def contact_modulus(self):
            if self.hidden:
                raise AttributeError('contact_modulus')
            return self.__dict__['_contact_modulus']

        @contact_modulus.setter
        def contact_modulus(self, value):
            self.__dict__['_contact_modulus'] = value

    reference = _periodic_system()
    assert getattr(reference, f'{kind}_minimize_proxy')(0.).success
    surface = reference.surface
    substrate = Substrate(surface.nb_grid_pts, 1., surface.physical_sizes, poisson=0.)
    substrate.hidden = True
    system = NonSmoothContactSystem(substrate, surface)
    assert getattr(system, f'{kind}_minimize_proxy')(0.).success
    np.testing.assert_allclose(system.force.sum(), reference.force.sum(), rtol=1e-6)
