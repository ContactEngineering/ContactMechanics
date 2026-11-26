"""
Tests of all modes of the `ce_hard_wall` command line tool, run in-process.
"""

import numpy as np
import pytest
from NuMPI import MPI
from SurfaceTopography import Topography

from ContactMechanics.CommandLineInterface.HardWall import main
from ContactMechanics.IO.NetCDF import NetCDFContainer

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial funcionalities, please execute with pytest",
)

# Columns of the log file
MEAN_PRESSURE, FRACTIONAL_AREA, CONVERGED = 4, 6, 7


@pytest.fixture
def sphere_fn(tmp_path):
    nb_grid_pts = 32
    x = (np.arange(nb_grid_pts) - nb_grid_pts / 2) / nb_grid_pts
    topography = Topography(-(x.reshape(-1, 1) ** 2 + x.reshape(1, -1) ** 2) / 2, (1., 1.))
    fn = str(tmp_path / 'sphere.nc')
    topography.to_netcdf(fn)
    return fn


def _run(tmp_path, topography_fn, *args):
    log_fn = str(tmp_path / 'log.out')
    main([topography_fn, '--log-fn', log_fn, '--contact-fn', str(tmp_path / 'contact'),
          '--pressure-fn', str(tmp_path / 'pressure'), '--gap-fn', str(tmp_path / 'gap'),
          '--displ-fn', str(tmp_path / 'displ'), '--netcdf-fn', str(tmp_path / 'out.nc')] + list(args))
    return np.loadtxt(log_fn, ndmin=2)


@pytest.mark.parametrize('boundary', ['periodic', 'nonperiodic'])
def test_automatic_steps(tmp_path, sphere_fn, boundary):
    log = _run(tmp_path, sphere_fn, '--nsteps', '4', '--boundary', boundary)
    assert len(log) == 4
    assert np.all(log[:, CONVERGED] == 1)
    # The contact areas span a range
    assert log[:, FRACTIONAL_AREA].min() < 0.01 < 0.1 < log[:, FRACTIONAL_AREA].max()
    for i in range(4):
        contact = np.loadtxt(tmp_path / f'contact.{i}')
        pressure = np.loadtxt(tmp_path / f'pressure.{i}')
        gap = np.loadtxt(tmp_path / f'gap.{i}')
        # The contact map contains the points with positive pressure
        np.testing.assert_array_equal(contact, pressure > 0)
        np.testing.assert_allclose(contact.mean(), log[i, FRACTIONAL_AREA])
        np.testing.assert_allclose(pressure.mean(), log[i, MEAN_PRESSURE])
        # Gap vanishes in the contact (up to the penetration tolerance)
        np.testing.assert_allclose(gap[contact == 1], 0, atol=1e-6)
    nc = NetCDFContainer(str(tmp_path / 'out.nc'))
    assert len(nc) == 4
    for i, frame in enumerate(nc):
        np.testing.assert_allclose(frame.load, log[i, MEAN_PRESSURE])
        np.testing.assert_allclose(frame.area, log[i, FRACTIONAL_AREA])
    nc.close()


def test_displacement_range(tmp_path, sphere_fn):
    # Also tests the syntax for values that start with a minus sign
    log = _run(tmp_path, sphere_fn, '--displacement=-0.005,0.015,3')
    assert len(log) == 3
    assert np.all(log[:, CONVERGED] == 1)
    # No contact at negative displacement, load increases with displacement
    assert log[0, MEAN_PRESSURE] == 0
    assert 0 < log[1, MEAN_PRESSURE] < log[2, MEAN_PRESSURE]
    # Rigid surface was displaced as requested
    np.testing.assert_allclose(np.diff(log[:, 1]), 0.01)


def test_single_displacement(tmp_path, sphere_fn):
    log = _run(tmp_path, sphere_fn, '--displacement', '0.005')
    assert len(log) == 1
    assert (tmp_path / 'contact').exists()


@pytest.mark.parametrize('from_file', [False, True])
def test_pressure(tmp_path, sphere_fn, from_file):
    pressures = [1e-3, 2e-3, 3e-3]
    if from_file:
        np.savetxt(tmp_path / 'pressures.txt', pressures)
        log = _run(tmp_path, sphere_fn, '--pressure-from-file', str(tmp_path / 'pressures.txt'))
    else:
        log = _run(tmp_path, sphere_fn, '--pressure', '1e-3,3e-3,3')
    assert np.all(log[:, CONVERGED] == 1)
    np.testing.assert_allclose(log[:, MEAN_PRESSURE], pressures)


def test_plastic(tmp_path, sphere_fn):
    hardness = 0.01
    log = _run(tmp_path, sphere_fn, '--nsteps', '3', '--hardness', str(hardness))
    assert np.all(log[:, CONVERGED] == 1)
    for i in range(3):
        contact = np.loadtxt(tmp_path / f'contact.{i}')
        pressure = np.loadtxt(tmp_path / f'pressure.{i}')
        assert pressure.max() <= hardness * (1 + 1e-12)
        # Plastically deformed points are part of the contact map
        np.testing.assert_array_equal(contact, pressure > 0)


def test_only_one_mode(tmp_path, sphere_fn):
    with pytest.raises(ValueError):
        _run(tmp_path, sphere_fn, '--pressure', '1e-3', '--displacement', '0.01')
