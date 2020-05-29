#
# Copyright 2019-2020 Lars Pastewka
#           2019 Antoine Sanner
# 
# ### MIT license
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

import pickle

import numpy as np
from numpy.testing import assert_array_equal

from muFFT import FFT
from NuMPI.Tools import Reduction

from PyCo.SurfaceTopography import Topography
from PyCo.SurfaceTopography.Generation import fourier_synthesis
from PyCo.SurfaceTopography.UniformLineScanAndTopography import DetrendedUniformTopography

from tests.Topography.PyCoTest import PyCoTestCase


def test_positions(comm):
    nx, ny = (12 * comm.Get_size(), 10 * comm.Get_size() + 1)
    sx = 33.
    sy = 54.
    fftengine = FFT((nx, ny), fft='mpi', communicator=comm)

    surf = Topography(np.zeros(fftengine.nb_subdomain_grid_pts),
                      physical_sizes=(sx, sy),
                      decomposition='subdomain',
                      nb_grid_pts=(nx, ny),
                      subdomain_locations=fftengine.subdomain_locations,
                      communicator=comm)

    x, y = surf.positions()
    assert x.shape == fftengine.nb_subdomain_grid_pts
    assert y.shape == fftengine.nb_subdomain_grid_pts

    assert Reduction(comm).min(x) == 0
    assert abs(Reduction(comm).max(x) - sx * (1 - 1. / nx)) < 1e-8 * sx / nx, "{}".format(x)
    assert Reduction(comm).min(y) == 0
    assert abs(Reduction(comm).max(y) - sy * (1 - 1. / ny)) < 1e-8


class TopographyTest(PyCoTestCase):

    def test_positions_and_heights(self):
        X = np.arange(3).reshape(1, 3)
        Y = np.arange(4).reshape(4, 1)
        h = X + Y

        t = Topography(h, (8, 6))

        self.assertEqual(t.nb_grid_pts, (4, 3))

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
        h = X + Y
        t = Topography(h, (8, 6))

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
        t = Topography(np.array([[1, 1, 1, 1],
                                 [1, 1, 1, 1],
                                 [1, 1, 1, 1]]), physical_sizes=(1, 1))

        # the following commands should be possible without errors
        st = t.scale(1)
        dt = st.detrend(detrend_mode='center')

    def test_power_spectrum_1D(self):
        X = np.arange(3).reshape(1, 3)
        Y = np.arange(4).reshape(4, 1)
        h = X + Y

        t = Topography(h, (8, 6))

        q1, C1 = t.power_spectrum_1D(window='hann')

        # TODO add check for values


def test_translate(comm_self):
    topography = Topography(np.array([[0, 1, 0], [0, 0, 0]]), physical_sizes=(4., 3.))
    print(topography.heights().shape)

    assert (topography.translate(offset=(1, 0)).heights()
            ==
            np.array([[0, 0, 0],
                      [0, 1, 0]])).all()

    assert (topography.translate(offset=(2, 0)).heights()
            ==
            np.array([[0, 1, 0],
                      [0, 0, 0]])).all()

    assert (topography.translate(offset=(0, -1)).heights()
            ==
            np.array([[1, 0, 0],
                      [0, 0, 0]])).all()


def test_pipeline():
    t1 = fourier_synthesis((511, 511), (1., 1.), 0.8, rms_height=1)
    t2 = t1.detrend()
    p = t2.pipeline()
    assert isinstance(p[0], Topography)
    assert isinstance(p[1], DetrendedUniformTopography)


def test_uniform_detrended_periodicity():
    topography = Topography(np.array([[0, 1, 0], [0, 0, 0]]), physical_sizes=(4., 3.), periodic=True)
    assert topography.detrend("center").is_periodic
    assert not topography.detrend("height").is_periodic
    assert not topography.detrend("curvature").is_periodic


def test_passing_of_docstring():
    from PyCo.SurfaceTopography.Uniform.PowerSpectrum import power_spectrum_1D
    topography = Topography(np.array([[0, 1, 0], [0, 0, 0]]), physical_sizes=(4., 3.), periodic=True)
    assert topography.power_spectrum_1D.__doc__ == power_spectrum_1D.__doc__