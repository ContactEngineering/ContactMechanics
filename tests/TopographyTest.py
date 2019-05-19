
import numpy as np
from numpy.testing import assert_array_equal
import pickle

from PyCo.Topography import Topography
from .PyCoTest import PyCoTestCase
from NuMPI.Tools import Reduction


def test_positions(comm, fftengine_class):
    nx, ny = (12*comm.Get_size(), 10 * comm.Get_size() +1)
    sx = 33.
    sy = 54.
    fftengine=fftengine_class((nx, ny), comm)
    pnp = Reduction(comm)

    surf = Topography(np.zeros(fftengine.subdomain_resolution), resolution=(nx, ny),
                      size = (sx, sy),
                      subdomain_location=fftengine.subdomain_location, pnp=pnp)

    x, y = surf.positions()
    assert x.shape == fftengine.subdomain_resolution
    assert y.shape == fftengine.subdomain_resolution

    assert pnp.min(x) == 0
    assert abs(pnp.max(x) - sx * (1-1./nx)) < 1e-8 * sx/ nx, "{}".format(x)
    assert pnp.min(y) == 0
    assert abs(pnp.max(y) - sy * (1-1./ny)) < 1e-8


class TopographyTest(PyCoTestCase):

    def test_positions_and_heights(self):

        X = np.arange(3).reshape(1, 3)
        Y = np.arange(4).reshape(4, 1)
        h = X+Y

        t = Topography(h, (8,6))

        self.assertEqual(t.resolution, (4,3))

        assert_array_equal(t.heights(), h)
        X2, Y2, h2 = t.positions_and_heights()
        assert_array_equal(X2, [
            (0, 0, 0),
            (2, 2, 2),
            (4, 4, 4),
            (6, 6, 6),
        ])
        assert_array_equal(Y2, [
            (0, 2, 4),
            (0, 2, 4),
            (0, 2, 4),
            (0, 2, 4),
        ])
        assert_array_equal(h2, [
            (0, 1, 2),
            (1, 2, 3),
            (2, 3, 4),
            (3, 4, 5)])

        #
        # After detrending, the position and heights should have again
        # just 3 arrays and the third array should be the same as .heights()
        #
        dt = t.detrend(detrend_mode='slope')

        self.assertArrayAlmostEqual(dt.heights(), [
            (0, 0, 0),
            (0, 0, 0),
            (0, 0, 0),
            (0, 0, 0)])

        X2, Y2, h2 = dt.positions_and_heights()

        assert h2.shape == (4, 3)
        assert_array_equal(X2, [
            (0, 0, 0),
            (2, 2, 2),
            (4, 4, 4),
            (6, 6, 6),
        ])
        assert_array_equal(Y2, [
            (0, 2, 4),
            (0, 2, 4),
            (0, 2, 4),
            (0, 2, 4),
        ])
        self.assertArrayAlmostEqual(h2, [
            (0, 0, 0),
            (0, 0, 0),
            (0, 0, 0),
            (0, 0, 0)])

    def test_squeeze(self):
        x = np.linspace(0, 4 * np.pi, 101)
        y = np.linspace(0, 8 * np.pi, 103)
        h = np.sin(x.reshape(-1, 1)) + np.cos(y.reshape(1, -1))
        surface = Topography(h, (1.2, 3.2)).scale(2.0)
        surface2 = surface.squeeze()
        self.assertTrue(isinstance(surface2, Topography))
        self.assertArrayAlmostEqual(surface.heights(), surface2.heights())

    def test_attribute_error(self):
        X = np.arange(3).reshape(1, 3)
        Y = np.arange(4).reshape(4, 1)
        h = X+Y
        t = Topography(h, (8,6))

        # nonsense attributes return attribute error
        with self.assertRaises(AttributeError):
            t.ababababababababa

        #
        # only scaled topographies have coeff
        #
        with self.assertRaises(AttributeError):
            t.coeff

        st = t.scale(1)

        self.assertEqual(st.scale_factor, 1)

        #
        # only detrended topographies have detrend_mode
        #
        with self.assertRaises(AttributeError):
            st.detrend_mode

        dm = st.detrend(detrend_mode='height').detrend_mode
        self.assertEqual(dm, 'height')

        #
        # this all should also work after pickling
        #
        t2 = pickle.loads(pickle.dumps(t))

        with self.assertRaises(AttributeError):
            t2.scale_factor

        st2 = t2.scale(1)

        self.assertEqual(st2.scale_factor, 1)

        with self.assertRaises(AttributeError):
            st2.detrend_mode

        dm2 = st2.detrend(detrend_mode='height').detrend_mode
        self.assertEqual(dm2, 'height')

        #
        # this all should also work after scaled+pickled
        #
        t3 = pickle.loads(pickle.dumps(st))

        with self.assertRaises(AttributeError):
            t3.detrend_mode

        dm3 = t3.detrend(detrend_mode='height').detrend_mode
        self.assertEqual(dm3, 'height')

    def test_init_with_lists_calling_scale_and_detrend(self):

        t = Topography([[1,1,1,1],
                        [1,1,1,1],
                        [1,1,1,1]], size=(1,1))

        # the following commands should be possible without errors
        st = t.scale(1)
        dt = st.detrend(detrend_mode='center')

    def test_power_spectrum_1D(self):

        X = np.arange(3).reshape(1, 3)
        Y = np.arange(4).reshape(4, 1)
        h = X+Y

        t = Topography(h, (8,6))

        q1, C1 = t.power_spectrum_1D(window='hann')

        # TODO add check for values