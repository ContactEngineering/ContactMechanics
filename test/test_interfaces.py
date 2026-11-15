"""
Tests of factory functions, argument validation and options of solvers,
substrates and Green's functions.
"""

import logging
import os
import tempfile
import uuid

import numpy as np
import pytest
from NuMPI import MPI
from NuMPI.Tools import Reduction
from SurfaceTopography import PlasticTopography, Topography, make_sphere

from ContactMechanics import ElasticSubstrate, FreeFFTElasticHalfSpace, PeriodicFFTElasticHalfSpace, make_system
from ContactMechanics.Factory import make_plastic_contact_system
from ContactMechanics.GreensFunctions import AnisotropicGreensFunction, IsotropicGreensFunction
from ContactMechanics.Optimization import constrained_conjugate_gradients
from ContactMechanics.PipelineFunction import _contact_calculation, _contact_steps
from ContactMechanics.PlasticSystemSpecialisations import PlasticNonSmoothContactSystem
from ContactMechanics.Tools.Logger import Logger


def _rough(periodic=True):
    rng = np.random.default_rng(1)
    # Smooth random heights
    filter_q = np.exp(-np.add.outer(np.fft.fftfreq(32) ** 2, np.fft.fftfreq(32) ** 2) * 100)
    heights = np.fft.ifft2(np.fft.fft2(rng.standard_normal((32, 32))) * filter_q).real
    return Topography(0.01 * heights / heights.std(), (1., 1.), periodic=periodic)


@pytest.mark.parametrize('periodic, substrate_class', [(True, PeriodicFFTElasticHalfSpace),
                                                       (False, FreeFFTElasticHalfSpace)])
def test_make_system_from_file(tmp_path, periodic, substrate_class):
    fn = str(tmp_path / 'topography.nc')
    _rough(periodic).to_netcdf(fn)
    # The substrate type is chosen from the periodicity of the topography.
    # (Every MPI process reads its own file, hence the serial communicator.)
    system = make_system(surface=fn, young=1., communicator=MPI.COMM_SELF)
    assert type(system.substrate) is substrate_class
    assert system.substrate.physical_sizes == (1., 1.)


def test_make_system_without_physical_sizes(tmp_path):
    fn = str(tmp_path / 'topography.npy')
    np.save(fn, np.zeros((8, 8)))
    with pytest.raises(ValueError):
        make_system(surface=fn, young=1., communicator=MPI.COMM_SELF)
    system = make_system(surface=fn, young=1., physical_sizes=(2., 2.), communicator=MPI.COMM_SELF,
                         substrate='periodic')
    assert system.substrate.physical_sizes == (2., 2.)


@pytest.mark.parametrize('substrate', ['periodic', 'free'])
def test_make_system_from_file_in_parallel(comm, substrate):
    # All processes read the same file; for the free substrate, some
    # processes only hold padding and have an empty topography subdomain
    token = comm.bcast(uuid.uuid4().hex if comm.rank == 0 else None, root=0)
    fn = os.path.join(tempfile.gettempdir(), f'contact_mechanics_{token}.npy')
    heights = np.arange(32 * 24, dtype=float).reshape(32, 24)
    if comm.rank == 0:
        np.save(fn, heights)
    comm.barrier()
    try:
        system = make_system(substrate=substrate, surface=fn, physical_sizes=(1., 1.), young=1., communicator=comm)
        local_heights = system.surface.heights()
        np.testing.assert_array_equal(local_heights, heights[system.substrate.topography_subdomain_slices])
        assert Reduction(comm).sum(local_heights.size) == heights.size
    finally:
        comm.barrier()
        if comm.rank == 0:
            os.remove(fn)


def test_make_plastic_contact_system():
    system = make_plastic_contact_system(PlasticTopography(_rough(), 0.1), young=1., communicator=None)
    assert isinstance(system, PlasticNonSmoothContactSystem)
    assert system.minimize_proxy(offset=0.).success


def test_pipeline_arguments(caplog):
    topography = _rough()
    for kwargs in [dict(), dict(nsteps=2, offsets=[0.], pressures=[0.1]), dict(nsteps=2, offsets=[0.]),
                   dict(nsteps=2, pressures=[0.1]), dict(offsets=[0.], pressures=[0.1])]:
        with pytest.raises(ValueError):
            topography.contact_mechanics(**kwargs)
    with pytest.raises(ValueError):
        next(_contact_steps(None))
    # Periodicity of topography and substrate differ
    with caplog.at_level(logging.WARNING):
        topography.contact_mechanics(substrate='nonperiodic', offsets=[0.])
    assert 'periodic' in caplog.text
    with caplog.at_level(logging.WARNING):
        _rough(periodic=False).contact_mechanics(substrate='periodic', offsets=[0.])
    assert 'not periodic' in caplog.text


def test_contact_calculation_with_pentol():
    system = make_system(substrate=PeriodicFFTElasticHalfSpace((32, 32), 1., (1., 1.)), surface=_rough())
    *_, history = _contact_calculation(system, offset=0., pentol=1e-10)
    assert history[-1][0]


def test_solver_options(tmp_path):
    topography = _rough()
    substrate = PeriodicFFTElasticHalfSpace((32, 32), 1., (1., 1.))
    reference = constrained_conjugate_gradients(substrate, topography, offset=0.)
    # Heights as array
    result = constrained_conjugate_gradients(substrate, topography.heights(), offset=0.)
    np.testing.assert_allclose(result.jac, reference.jac)
    # Initial forces
    result = constrained_conjugate_gradients(substrate, topography, offset=0.,
                                             initial_forces=-reference.jac)
    assert result.success
    assert result.nit <= reference.nit
    np.testing.assert_allclose(result.jac.sum(), reference.jac.sum(), rtol=1e-6)
    # Verbose logging, also in the mixing regime of plastic calculations
    fn = tmp_path / 'log.out'
    constrained_conjugate_gradients(substrate, topography, offset=0., hardness=1e-12, logger=Logger(str(fn)),
                                    verbose=True, maxiter=50)
    log = fn.read_text()
    assert 'mixfac' in log and 'tau' in log


def test_substrate_base_class():
    class Minimal(ElasticSubstrate):
        nb_domain_grid_pts = nb_subdomain_grid_pts = subdomain_locations = communicator = None

        def evaluate(self, disp, pot=True, forces=False):
            return super().evaluate(disp, pot, forces)

    substrate = Minimal((4, 2), 2.)
    assert substrate.physical_sizes == (2., 2.)
    assert 'x: 2.0(4)' in repr(substrate)
    with pytest.raises(ElasticSubstrate.Error):
        substrate.spawn_child((2, 1))
    with pytest.raises(ElasticSubstrate.Error):
        Minimal.is_periodic()
    with pytest.raises(NotImplementedError):
        substrate.evaluate(None)
    with pytest.raises(AttributeError):
        substrate.compute_k(None)
    substrate.check()
    assert PeriodicFFTElasticHalfSpace.is_periodic()
    assert not FreeFFTElasticHalfSpace.is_periodic()


def test_isotropic_arrays():
    gf = IsotropicGreensFunction(1.0, 0.3)
    qx = np.array([1.0, 0.3])
    qy = np.array([0.0, 0.8])
    stiffness = gf.stiffness(qx, qy)
    compliance = gf.greens_function(qx, qy)
    for i in range(2):
        np.testing.assert_allclose(stiffness[i], gf.stiffness(qx[i], qy[i]))
        np.testing.assert_allclose(stiffness[i] @ compliance[i], np.eye(3), atol=1e-12)


def test_anisotropic_gamma_point():
    with pytest.raises(ValueError):
        AnisotropicGreensFunction(1.0, 0.5, 0.3, R=-np.eye(3))
    thickness = 2.0
    gf = AnisotropicGreensFunction(1.0, 0.5, 0.3, thickness=thickness)
    stiffness = gf.stiffness(0.0, 0.0)
    # Uniform deformation of a layer: shear with C44, compression with C11
    np.testing.assert_allclose(stiffness, np.diag([0.3, 0.3, 1.0]) / thickness)
    np.testing.assert_allclose(gf.greens_function(0.0, 0.0), np.linalg.inv(stiffness))
    np.testing.assert_array_equal(AnisotropicGreensFunction(1.0, 0.5, 0.3).stiffness(0.0, 0.0), 0)


def test_plastic_system_handles_sphere():
    surface = PlasticTopography(make_sphere(1., (16, 16), (1., 1.), centre=(0.5, 0.5), kind='paraboloid'), 0.05)
    system = PlasticNonSmoothContactSystem(FreeFFTElasticHalfSpace((16, 16), 1., (1., 1.)), surface)
    assert system.minimize_proxy(offset=0.01).success
    assert surface.plastic_area > 0
