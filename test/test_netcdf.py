"""
Tests of the NetCDF container for contact calculations. The tests work with
both backends (netCDF4 and scipy).
"""

import numpy as np
import pytest
from scipy.io import netcdf_file

from ContactMechanics.IO import NetCDF
from ContactMechanics.IO.NetCDF import NetCDFContainer


@pytest.fixture
def container_fn(tmp_path):
    """File with two frames, a rigid surface and container variables"""
    fn = str(tmp_path / 'container.nc')
    rng = np.random.default_rng(0)
    data = dict(h=rng.random((4, 3)), elastic=rng.random((4, 3)), mu=rng.random((4, 3)),
                u=[rng.random((4, 3)), rng.random((4, 3))], partial=rng.random((2, 2)))

    nc = NetCDFContainer(fn, mode='w', double=True)
    nc.set_shape((4, 3))
    nc.set_rigid_surface(data['h'])
    nc.set_elastic_surface(data['elastic'])
    nc.mu = data['mu']
    nc.title = 'test'
    for i in range(2):
        frame = nc.get_next_frame()
        frame.displacements = data['u'][i]
        frame.load = 1.5 * i
        frame.step = i
        frame['area'] = 0.25 * i
        frame.set_grid('partial', data['partial'], x0=1, y0=1)
    nc.sync()
    nc.close()
    return fn, data


def test_read(container_fn):
    fn, data = container_fn
    nc = NetCDFContainer(fn)
    assert len(nc) == 2
    assert nc.get_size() == (4, 3)
    assert nc.has_h()
    assert str(nc) == fn
    assert nc.get_filename() == fn
    np.testing.assert_allclose(nc.get_rigid_surface(), data['h'])
    np.testing.assert_allclose(nc.get_h(), data['h'])
    np.testing.assert_allclose(nc.get_elastic_surface()[...], data['elastic'])
    np.testing.assert_allclose(nc['mu'], data['mu'])
    np.testing.assert_allclose(nc.mu, data['mu'])
    assert nc.title in ('test', b'test')
    for i, frame in enumerate(nc):
        assert frame.index == i
        assert frame.traj is nc
        np.testing.assert_allclose(frame.displacements, data['u'][i])
        np.testing.assert_allclose(frame['load'], 1.5 * i)
        assert frame.step == i
        np.testing.assert_allclose(frame.area, 0.25 * i)
        np.testing.assert_allclose(frame.partial[1:3, 1:3], data['partial'])
    # Negative and slice indices
    assert nc[-1].index == 1
    assert [frame.index for frame in nc[0:2]] == [0, 1]
    nc.close()


def test_cursor(container_fn):
    fn, data = container_fn
    nc = NetCDFContainer(fn, frame=-1)
    assert nc.get_cursor() == 1
    nc.set_cursor(0)
    np.testing.assert_allclose(nc.get_next_frame().displacements, data['u'][0])
    assert nc.get_cursor() == 1
    nc.close()


def test_open(container_fn):
    fn, data = container_fn
    nc = NetCDF.open(fn)
    assert isinstance(nc, NetCDFContainer)
    assert NetCDF.open(nc) is nc
    np.testing.assert_allclose(NetCDF.open(fn + '@1').displacements, data['u'][1])
    np.testing.assert_allclose(NetCDF.open(fn, frame=0).displacements, data['u'][0])
    nc.close()


def test_rigid_surface_of_different_size(tmp_path):
    fn = str(tmp_path / 'rigid.nc')
    nc = NetCDFContainer(fn, mode='w')
    nc.set_shape((4, 3))
    h = np.arange(10.).reshape(5, 2)
    nc.set_rigid_surface(h)
    nc.close()
    nc = NetCDFContainer(fn)
    np.testing.assert_allclose(nc.get_rigid_surface(), h)
    nc.close()


def test_multiple_degrees_of_freedom(tmp_path):
    fn = str(tmp_path / 'ndof.nc')
    nc = NetCDFContainer(fn, mode='w')
    nc.set_shape((4, 3), ndof=3)
    nc.set_shape((4, 3), ndof=3)  # Consistent shape is fine
    u = np.arange(36.).reshape(3, 4, 3)
    nc.get_next_frame().u = u
    nc.close()
    nc = NetCDFContainer(fn)
    np.testing.assert_allclose(nc[0].u, u)
    nc.close()


def test_errors(tmp_path):
    fn = str(tmp_path / 'errors.nc')
    nc = NetCDFContainer(fn, mode='w')
    nc.set_shape((4, 3))
    with pytest.raises(RuntimeError):
        nc.set_shape((5, 3))
    frame = nc.get_next_frame()
    with pytest.raises(RuntimeError):
        frame.wrong_shape = np.zeros((7, 7))
    with pytest.raises(RuntimeError):
        frame.wrong_type = 'text'
    with pytest.raises(RuntimeError):
        nc[0] = np.zeros((4, 3))
    with pytest.raises(RuntimeError):
        nc.wrong_shape = np.zeros((7, 7))
    nc.close()


def test_unknown_convention(tmp_path):
    fn = str(tmp_path / 'unknown.nc')
    f = netcdf_file(fn, 'w')
    f.createDimension('foo', 3)
    f.createVariable('bar', 'f8', ('foo',))[:] = np.arange(3.)
    f.close()
    with pytest.raises(RuntimeError):
        NetCDFContainer(fn)


def test_amber_convention(tmp_path):
    # Files following the AMBER convention store coordinates per atom
    fn = str(tmp_path / 'amber.nc')
    coordinates = np.arange(12., dtype=np.float32).reshape(1, 4, 3)
    f = netcdf_file(fn, 'w')
    f.createDimension('frame', None)
    f.createDimension('atom', 4)
    f.createDimension('spatial', 3)
    f.createVariable('coordinates', 'f4', ('frame', 'atom', 'spatial'))[0] = coordinates[0]
    f.close()
    nc = NetCDFContainer(fn)
    assert nc.get_size() == (2, 2)
    np.testing.assert_allclose(nc[0].u, coordinates[0].reshape(2, 2, 3))
    nc.close()
