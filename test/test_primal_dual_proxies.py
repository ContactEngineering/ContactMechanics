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
from SurfaceTopography import Topography, make_sphere

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
