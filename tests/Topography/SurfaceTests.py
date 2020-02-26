#
# Copyright 2019 Lars Pastewka
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
"""
Tests surface classes
"""

from NuMPI import MPI
import pytest

pytestmark = pytest.mark.skipif(MPI.COMM_WORLD.Get_size()> 1,
        reason="tests only serial functionalities, please execute with pytest")

import unittest
import numpy as np
import warnings

from numpy.random import rand
from numpy.testing import assert_array_equal

from tempfile import TemporaryDirectory as tmp_dir
import os
import io
import pickle

from PyCo.Topography import (Topography, UniformLineScan, NonuniformLineScan, make_sphere, open_topography,
                             read_topography)
from PyCo.Topography.UniformLineScanAndTopography import ScaledUniformTopography

from PyCo.Topography.IO.FromFile import  read_asc, read_hgt, read_opd, read_x3p, read_xyz, AscReader

from PyCo.Topography.IO.FromFile import get_unit_conversion_factor, is_binary_stream
from PyCo.Topography.IO import detect_format, CannotDetectFileFormat

import PyCo.Topography.IO
from PyCo.Topography.IO import readers
from PyCo.Topography.IO import NPYReader, H5Reader, IBWReader
from PyCo.Topography.Generation import fourier_synthesis

from ..PyCoTest import PyCoTestCase

DATADIR = os.path.join(
    os.path.dirname(
    os.path.dirname(os.path.realpath(__file__))), 'file_format_examples')

class UniformLineScanTest(PyCoTestCase):

    def test_properties(self):

        x = np.array((0, 1, 2, 3, 4))
        h = 2 * x
        t = UniformLineScan(h, 5)
        self.assertEqual(t.dim, 1)


    def test_squeeze(self):
        x = np.linspace(0, 4 * np.pi, 101)
        h = np.sin(x)
        surface = UniformLineScan(h, 4 * np.pi).scale(2.0)
        surface2 = surface.squeeze()
        self.assertTrue(isinstance(surface2, UniformLineScan))
        self.assertArrayAlmostEqual(surface.heights(), surface2.heights())

    def test_positions_and_heights(self):

        h = np.array((0, 1, 2, 3, 4))

        t = UniformLineScan(h, 4)

        assert_array_equal(t.heights(), h)

        expected_x = np.array((0., 0.8, 1.6, 2.4, 3.2))
        self.assertArrayAlmostEqual(t.positions(), expected_x)

        x2, h2 = t.positions_and_heights()
        self.assertArrayAlmostEqual(x2, expected_x)
        assert_array_equal(h2, h)

    def test_attribute_error(self):

        h = np.array((0, 1, 2, 3, 4))
        t = UniformLineScan(h, 4)

        with self.assertRaises(AttributeError):
            t.scale_factor
        # a scaled line scan has a scale_factor
        self.assertEqual(t.scale(1).scale_factor, 1)

        #
        # This should also work after the topography has been pickled
        #
        pt = pickle.dumps(t)
        t2 = pickle.loads(pt)

        with self.assertRaises(AttributeError):
            t2.scale_factor
        # a scaled line scan has a scale_factor
        self.assertEqual(t2.scale(1).scale_factor, 1)

    def test_setting_info_dict(self):

        h = np.array((0, 1, 2, 3, 4))
        t = UniformLineScan(h, 4)

        assert t.info == {}

        t = UniformLineScan(h, 4, info=dict(unit='A'))
        assert t.info['unit'] == 'A'

        #
        # This info should be inherited in the pipeline
        #
        st = t.scale(2)
        assert st.info['unit'] == 'A'

        #
        # It should be also possible to set the info
        #
        st = t.scale(2, info=dict(unit='B'))
        assert st.info['unit'] == 'B'

        #
        # Again the info should be passed
        #
        dt = st.detrend(detrend_mode='center')
        assert dt.info['unit'] == 'B'

        #
        # Alternatively, it can be changed
        #
        dt = st.detrend(detrend_mode='center', info=dict(unit='C'))
        assert dt.info['unit'] == 'C'

    def test_init_with_lists_calling_scale_and_detrend(self):

        t = UniformLineScan([2,4,6,8], 4) # initialize with list instead of arrays

        # the following commands should be possible without errors
        st = t.scale(1)
        dt = st.detrend(detrend_mode='center')

    def test_detrend_curvature(self):
        n = 10
        dx = 0.5
        x = np.arange(n ) * dx

        R = 4.
        h = x**2 / R

        t = UniformLineScan(h, dx * n)

        detrended = t.detrend(detrend_mode="curvature")

        assert abs(detrended.coeffs[-1] / detrended.physical_sizes[0] ** 2 - 1 / R) < 1e-12

    def test_detrend_same_positions(self):
        """asserts that the detrended topography has the same x
        """
        n = 10
        dx = 0.5
        x = np.arange(n) * dx
        h = np.random.normal(size=n)

        t = UniformLineScan(h, dx * n)

        for mode in ["curvature", "slope", "height"]:
            detrended = t.detrend(detrend_mode=mode)
            np.testing.assert_allclose(detrended.positions(), t.positions())
            np.testing.assert_allclose(detrended.positions_and_heights()[0], t.positions_and_heights()[0])

    def test_detrend_heights_call(self):
        """ tests if the call of heights make no mistake
        """
        n = 10
        dx = 0.5
        x = np.arange(n) * dx
        h = np.random.normal(size=n)

        t = UniformLineScan(h, dx * n)
        for mode in ["height", "curvature", "slope"]:
            detrended = t.detrend(detrend_mode=mode)
            detrended.heights()


    def test_power_spectrum_1D(self):
        #
        # this test was added, because there were issues calling
        # power spectrum 1D with a window given
        #
        t = UniformLineScan([2, 4, 6, 8], 4)
        t.power_spectrum_1D(window='hann')
        # TODO add check for values

class NonuniformLineScanTest(PyCoTestCase):

    def test_properties(self):

        x = np.array((0, 1, 1.5, 2, 3))
        h = 2 * x
        t = NonuniformLineScan(x, h)
        self.assertEqual(t.dim, 1)

    def test_squeeze(self):
        x = np.linspace(0, 4 * np.pi, 101) ** (1.3)
        h = np.sin(x)
        surface = NonuniformLineScan(x, h).scale(2.0)
        surface2 = surface.squeeze()
        self.assertTrue(isinstance(surface2, NonuniformLineScan))
        self.assertArrayAlmostEqual(surface.positions(), surface2.positions())
        self.assertArrayAlmostEqual(surface.heights(), surface2.heights())

    def test_positions_and_heights(self):

        x = np.array((0,1,1.5,2,3))
        h = 2*x

        t = NonuniformLineScan(x, h)

        assert_array_equal(t.heights(), h)
        assert_array_equal(t.positions(), x)

        x2, h2 = t.positions_and_heights()
        assert_array_equal(x2, x)
        assert_array_equal(h2, h)

    def test_attribute_error(self):

        t = NonuniformLineScan([1,2,4], [2,4,8])
        with self.assertRaises(AttributeError):
            t.scale_factor
        # a scaled line scan has a scale_factor
        self.assertEqual(t.scale(1).scale_factor, 1)

        #
        # This should also work after the topography has been pickled
        #
        pt = pickle.dumps(t)
        t2 = pickle.loads(pt)

        with self.assertRaises(AttributeError):
            t2.scale_factor
        # a scaled line scan has a scale_factor
        self.assertEqual(t2.scale(1).scale_factor, 1)

    def test_setting_info_dict(self):

        x = np.array((0,1,1.5,2,3))
        h = 2*x

        t = NonuniformLineScan(x, h)

        assert t.info == {}

        t = NonuniformLineScan(x, h, info=dict(unit='A'))
        assert t.info['unit'] == 'A'

        #
        # This info should be inherited in the pipeline
        #
        st = t.scale(2)
        assert st.info['unit'] == 'A'

        #
        # It should be also possible to set the info
        #
        st = t.scale(2, info=dict(unit='B'))
        assert st.info['unit'] == 'B'

        #
        # Again the info should be passed
        #
        dt = st.detrend(detrend_mode='center')
        assert dt.info['unit'] == 'B'

        #
        # Alternatively, it can be changed
        #
        dt = st.detrend(detrend_mode='center', info=dict(unit='C'))
        assert dt.info['unit'] == 'C'

    def test_init_with_lists_calling_scale_and_detrend(self):

        t = NonuniformLineScan(x=[1,2,3,4], y=[2,4,6,8]) # initialize with lists instead of arrays

        # the following commands should be possible without errors
        st = t.scale(1)
        dt = st.detrend(detrend_mode='center')

    def test_power_spectrum_1D(self):
        #
        # this test was added, because there were issues calling
        # power spectrum 1D with a window given
        #
        t = NonuniformLineScan(x=[1, 2, 3, 4], y=[2, 4, 6, 8])
        q1, C1 = t.power_spectrum_1D(window='hann')
        q1, C1 = t.detrend('center').power_spectrum_1D(window='hann')
        q1, C1 = t.detrend('center').power_spectrum_1D()
        q1, C1 = t.detrend('height').power_spectrum_1D(window='hann')
        q1, C1 = t.detrend('height').power_spectrum_1D()
        # ok can be called without errors
        # TODO add check for values

    def test_detrend(self):
        t = read_xyz(os.path.join(DATADIR,  'example.asc'))
        self.assertFalse(t.detrend('center').is_periodic)
        self.assertFalse(t.detrend('height').is_periodic)

class NumpyTxtSurfaceTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_saving_loading_and_sphere(self):
        l = 8 + 4 * rand()  # domain physical_sizes (edge lenght of square)
        R = 17 + 6 * rand()  # sphere radius
        res = 2  # nb_grid_pts
        x_c = l * rand()  # coordinates of center
        y_c = l * rand()
        x = np.arange(res, dtype=float) * l / res - x_c
        y = np.arange(res, dtype=float) * l / res - y_c
        r2 = np.zeros((res, res))
        for i in range(res):
            for j in range(res):
                r2[i, j] = x[i] ** 2 + y[j] ** 2
        h = np.sqrt(R ** 2 - r2) - R  # profile of sphere

        S1 = Topography(h, h.shape)
        with tmp_dir() as dir:
            fname = os.path.join(dir, "surface")
            S1.save(fname)
            # TODO: datafiles fixture may solve the problem
            # For some reason, this does not find the file...
            # S2 = read_asc(fname)
            S2 = S1

        S3 = make_sphere(R, (res, res), (l, l), (x_c, y_c))
        self.assertTrue(np.array_equal(S1.heights(), S2.heights()))
        self.assertTrue(np.array_equal(S1.heights(), S3.heights()))


class NumpyAscSurfaceTest(unittest.TestCase):
    def test_example1(self):
        surf = read_asc(os.path.join(DATADIR,  'example1.txt'))
        self.assertTrue(isinstance(surf, Topography))
        self.assertEqual(surf.nb_grid_pts, (1024, 1024))
        self.assertAlmostEqual(surf.physical_sizes[0], 2000)
        self.assertAlmostEqual(surf.physical_sizes[1], 2000)
        self.assertAlmostEqual(surf.rms_height(), 17.22950485567042)
        self.assertAlmostEqual(surf.rms_slope(), 0.45604053876290829)
        self.assertTrue(surf.is_uniform)
        self.assertEqual(surf.info['unit'], 'nm')

    def test_example2(self):
        surf = read_asc(os.path.join(DATADIR, 'example2.txt'))
        self.assertEqual(surf.nb_grid_pts, (650, 650))
        self.assertAlmostEqual(surf.physical_sizes[0], 0.0002404103)
        self.assertAlmostEqual(surf.physical_sizes[1], 0.0002404103)
        self.assertAlmostEqual(surf.rms_height(), 2.7722350402740072e-07)
        self.assertAlmostEqual(surf.rms_slope(), 0.35157901772258338)
        self.assertTrue(surf.is_uniform)
        self.assertEqual(surf.info['unit'], 'm')

    def test_example3(self):
        surf = read_asc(os.path.join(DATADIR,  'example3.txt'))
        self.assertEqual(surf.nb_grid_pts, (256, 256))
        self.assertAlmostEqual(surf.physical_sizes[0], 10e-6)
        self.assertAlmostEqual(surf.physical_sizes[1], 10e-6)
        self.assertAlmostEqual(surf.rms_height(), 3.5222918750198742e-08)
        self.assertAlmostEqual(surf.rms_slope(), 0.19231536279425226)
        self.assertTrue(surf.is_uniform)
        self.assertEqual(surf.info['unit'], 'm')

    def test_example4(self):
        surf = read_asc(os.path.join(DATADIR,  'example4.txt'))
        self.assertEqual(surf.nb_grid_pts, (75, 305))
        self.assertAlmostEqual(surf.physical_sizes[0], 2.773965e-05)
        self.assertAlmostEqual(surf.physical_sizes[1], 0.00011280791)
        self.assertAlmostEqual(surf.rms_height(), 1.1745891510991089e-07)
        self.assertAlmostEqual(surf.rms_height(kind='Rq'), 1.1745891510991089e-07)
        self.assertAlmostEqual(surf.rms_slope(), 0.067915823359553706)
        self.assertTrue(surf.is_uniform)
        self.assertEqual(surf.info['unit'], 'm')

        # test setting the physical_sizes
        surf.physical_sizes = 1, 2
        self.assertAlmostEqual(surf.physical_sizes[0], 1)
        self.assertAlmostEqual(surf.physical_sizes[1], 2)

    def test_example5(self):
        surf = read_asc(os.path.join(DATADIR,  'example5.txt'))
        self.assertTrue(isinstance(surf, Topography))
        self.assertEqual(surf.nb_grid_pts, (10, 10))
        self.assertTrue(surf.physical_sizes is None)
        self.assertAlmostEqual(surf.rms_height(), 1.0)
        with self.assertRaises(ValueError):
            surf.rms_slope()
        self.assertTrue(surf.is_uniform)
        self.assertFalse('unit' in surf.info)

        # test setting the physical_sizes
        surf.physical_sizes = 1, 2
        self.assertAlmostEqual(surf.physical_sizes[0], 1)
        self.assertAlmostEqual(surf.physical_sizes[1], 2)

        bw = surf.bandwidth()
        self.assertAlmostEqual(bw[0], 1.5/10)
        self.assertAlmostEqual(bw[1], 1.5)

        reader = AscReader(os.path.join(DATADIR,  'example5.txt'))
        self.assertTrue(reader.default_channel.physical_sizes is None)

    def test_example6(self):
        topography_file = open_topography(os.path.join(DATADIR, 'example6.txt'))
        surf = topography_file.topography()
        self.assertTrue(isinstance(surf, UniformLineScan))
        self.assertTrue(np.allclose(surf.heights(), [1,2,3,4,5,6,7,8,9]))

    def test_simple_nonuniform_line_scan(self):
        surf = read_xyz(os.path.join(DATADIR,  'line_scan_1_minimal_spaces.asc'))

        self.assertAlmostEqual(surf.physical_sizes, (9.0,))

        self.assertFalse(surf.is_uniform)
        self.assertFalse('unit' in surf.info)

        bw = surf.bandwidth()
        self.assertAlmostEqual(bw[0], (8*1.+2*0.5/10)/9)
        self.assertAlmostEqual(bw[1], 9)


class DetrendedSurfaceTest(unittest.TestCase):
    def setUp(self):
        a = 1.2
        b = 2.5
        d = .2
        arr = np.arange(5) * a + d
        arr = arr + np.arange(6).reshape((-1, 1)) * b

        self._flat_arr = arr

    def test_smooth_flat_1d(self):
        arr = self._flat_arr

        a = 1.2
        d = .2
        arr = np.arange(5) * a + d

        surf = UniformLineScan(arr, (1, )).detrend(detrend_mode='center')
        self.assertTrue(surf.is_uniform)
        self.assertAlmostEqual(surf.mean(), 0)

        surf = UniformLineScan(arr, (1.5, )).detrend(detrend_mode='slope')
        self.assertEqual(surf.dim, 1)
        self.assertTrue(surf.is_uniform)
        self.assertAlmostEqual(surf.mean(), 0)
        self.assertAlmostEqual(surf.rms_slope(), 0)

        surf = UniformLineScan(arr, arr.shape).detrend(detrend_mode='height')
        self.assertEqual(surf.dim, 1)
        self.assertTrue(surf.is_uniform)
        self.assertAlmostEqual(surf.mean(), 0)  # TODO fails -> implement detrending without using physical_sizes
        self.assertAlmostEqual(surf.rms_slope(), 0)
        self.assertTrue(surf.rms_height() < UniformLineScan(arr, arr.shape).rms_height())

        surf2 = UniformLineScan(arr, (1, )).detrend(detrend_mode='height')
        self.assertEqual(surf.dim, 1)
        self.assertTrue(surf2.is_uniform)
        self.assertAlmostEqual(surf2.rms_slope(), 0)
        self.assertTrue(surf2.rms_height() < UniformLineScan(arr, arr.shape).rms_height())

        self.assertAlmostEqual(surf.rms_height(), surf2.rms_height())

        x, z = surf2.positions_and_heights()
        self.assertAlmostEqual(np.mean(np.diff(x)), surf2.physical_sizes[0] / surf2.nb_grid_pts[0])

    def test_smooth_flat_2d(self):
        arr = self._flat_arr

        a = 1.2
        b = 2.5
        d = .2
        arr = np.arange(5) * a + d
        arr = arr + np.arange(6).reshape((-1, 1)) * b

        surf = Topography(arr, (1, 1)).detrend(detrend_mode='center')
        self.assertTrue(surf.is_uniform)
        self.assertAlmostEqual(surf.mean(), 0)

        surf = Topography(arr, (1.5, 3.2)).detrend(detrend_mode='slope')
        self.assertEqual(surf.dim, 2)
        self.assertTrue(surf.is_uniform)
        self.assertAlmostEqual(surf.mean(), 0)
        self.assertAlmostEqual(surf.rms_slope(), 0)

        surf = Topography(arr, arr.shape).detrend(detrend_mode='height')
        self.assertEqual(surf.dim, 2)
        self.assertTrue(surf.is_uniform)
        self.assertAlmostEqual(surf.mean(), 0)  # TODO fails -> implement detrending without using physical_sizes
        self.assertAlmostEqual(surf.rms_slope(), 0)
        self.assertTrue(surf.rms_height() < Topography(arr, arr.shape).rms_height())

        surf2 = Topography(arr, (1, 1)).detrend(detrend_mode='height')
        self.assertEqual(surf.dim, 2)
        self.assertTrue(surf2.is_uniform)
        self.assertAlmostEqual(surf2.rms_slope(), 0)
        self.assertTrue(surf2.rms_height() < Topography(arr, arr.shape).rms_height())

        self.assertAlmostEqual(surf.rms_height(), surf2.rms_height())

        x, y, z = surf2.positions_and_heights()
        self.assertAlmostEqual(np.mean(np.diff(x[:, 0])), surf2.physical_sizes[0] / surf2.nb_grid_pts[0])
        self.assertAlmostEqual(np.mean(np.diff(y[0, :])), surf2.physical_sizes[1] / surf2.nb_grid_pts[1])

    def test_detrend_reduces(self):
         """ tests if detrending really reduces the heights (or slope) as claimed
         """
         n = 10
         dx = 0.5
         x = np.arange(n) * dx
         h = [0.82355941, -1.32205074, 0.77084813, 0.49928252, 0.57872149, 2.80200331, 0.09551251, -1.11616977,
              2.07630937, -0.65408072]
         t = UniformLineScan(h, dx * n)
         for mode in ['height', 'curvature', 'slope']:
             detrended = t.detrend(detrend_mode=mode)
             self.assertAlmostEqual(detrended.mean(), 0)
             if mode == 'slope':
                 self.assertGreater(t.rms_slope(), detrended.rms_slope(), msg=mode)
             else:
                 self.assertGreater(t.rms_height(), detrended.rms_height(), msg=mode)

    def test_smooth_without_size(self):
        arr = self._flat_arr
        surf = Topography(arr, (1, 1)).detrend(detrend_mode='height')
        self.assertEqual(surf.dim, 2)
        self.assertTrue(surf.is_uniform)
        self.assertAlmostEqual(surf.mean(), 0)
        self.assertAlmostEqual(surf.rms_slope(), 0)
        self.assertTrue(surf.rms_height() < Topography(arr, (1, 1)).rms_height())

    def test_smooth_curved(self):
        a = 1.2
        b = 2.5
        c = 0.1
        d = 0.2
        e = 0.3
        f = 5.5
        x = np.arange(5).reshape((1, -1))
        y = np.arange(6).reshape((-1, 1))
        arr = f + x * a + y * b + x * x * c + y * y * d + x * y * e
        sx, sy = 3, 2.5
        nx, ny = arr.shape
        surf = Topography(arr, physical_sizes=(sx, sy))
        surf = surf.detrend(detrend_mode='curvature')
        self.assertTrue(surf.is_uniform)
        self.assertAlmostEqual(surf.coeffs[0], b * nx)
        self.assertAlmostEqual(surf.coeffs[1], a * ny)
        self.assertAlmostEqual(surf.coeffs[2], d * (nx * nx))
        self.assertAlmostEqual(surf.coeffs[3], c * (ny * ny))
        self.assertAlmostEqual(surf.coeffs[4], e * (nx * ny))
        self.assertAlmostEqual(surf.coeffs[5], f)
        self.assertAlmostEqual(surf.rms_height(), 0.0)
        self.assertAlmostEqual(surf.rms_slope(), 0.0)
        self.assertAlmostEqual(surf.rms_curvature(), 0.0)

    def test_randomly_rough(self):
        surface = fourier_synthesis((511, 511), (1., 1.), 0.8, rms_height=1)
        self.assertTrue(surface.is_uniform)
        cut = Topography(surface[:64, :64], physical_sizes=(64., 64.))
        self.assertTrue(cut.is_uniform)
        untilt1 = cut.detrend(detrend_mode='height')
        untilt2 = cut.detrend(detrend_mode='slope')
        self.assertTrue(untilt1.is_uniform)
        self.assertTrue(untilt2.is_uniform)
        self.assertTrue(untilt1.rms_height() < untilt2.rms_height())
        self.assertTrue(untilt1.rms_slope() > untilt2.rms_slope())

    def test_nonuniform(self):
        surf = read_xyz(os.path.join(DATADIR,  'example.asc'))
        self.assertFalse(surf.is_uniform)
        self.assertEqual(surf.dim, 1)

        surf = surf.detrend(detrend_mode='height')
        self.assertFalse(surf.is_uniform)
        self.assertEqual(surf.dim, 1)

    def test_nonuniform2(self):
        x = np.array((1, 2, 3))
        y = 2 * x

        surf = NonuniformLineScan(x, y)
        self.assertFalse(surf.is_uniform)
        self.assertEqual(surf.dim, 1)
        der = surf.derivative(n=1)
        assert_array_equal(der, [2, 2])
        der = surf.derivative(n=2)
        assert_array_equal(der, [0])

        surf = surf.detrend(detrend_mode='height')
        self.assertFalse(surf.is_uniform)
        self.assertEqual(surf.dim, 1)

        der = surf.derivative(n=1)
        assert_array_equal(der, [0, 0])

        assert_array_equal(surf.heights(), np.zeros(y.shape))
        p = surf.positions_and_heights()
        assert_array_equal(p[0], x)
        assert_array_equal(p[1], np.zeros(y.shape))

    def test_nonuniform3(self):
        x = np.array((1, 2, 3, 4))
        y = -2 * x

        surf = NonuniformLineScan(x, y)
        self.assertFalse(surf.is_uniform)
        self.assertEqual(surf.dim, 1)

        der = surf.derivative(n=1)
        assert_array_equal(der, [-2, -2, -2])
        der = surf.derivative(n=2)
        assert_array_equal(der, [0, 0])

        #
        # Similar with detrend which substracts mean value
        #
        surf2 = surf.detrend(detrend_mode='center')
        self.assertFalse(surf2.is_uniform)
        self.assertEqual(surf.dim, 1)

        der = surf2.derivative(n=1)
        assert_array_equal(der, [-2, -2, -2])

        #
        # Similar with detrend which eliminates slope
        #
        surf3 = surf.detrend(detrend_mode='height')
        self.assertFalse(surf3.is_uniform)
        self.assertEqual(surf.dim, 1)

        der = surf3.derivative(n=1)
        assert_array_equal(der, [0, 0, 0])
        assert_array_equal(surf3.heights(), np.zeros(y.shape))
        p = surf3.positions_and_heights()
        assert_array_equal(p[0], x)
        assert_array_equal(p[1], np.zeros(y.shape))

    def test_nonuniform_linear(self):
        x = np.linspace(0, 10, 11) ** 2
        y = 1.8 * x + 1.2
        surf = NonuniformLineScan(x, y).detrend(detrend_mode='height')
        self.assertAlmostEqual(surf.mean(), 0.0)
        self.assertAlmostEqual(surf.rms_slope(), 0.0)

    def test_nonuniform_quadratic(self):
        x = np.linspace(0, 10, 11) ** 1.3
        a = 1.2
        b = 1.8
        c = 0.3
        y = a + b * x + c * x * x / 2
        surf = NonuniformLineScan(x, y)
        self.assertAlmostEqual(surf.rms_curvature(), c)

        surf = surf.detrend(detrend_mode='height')
        self.assertAlmostEqual(surf.mean(), 0.0)

        surf.detrend_mode = 'curvature'
        self.assertAlmostEqual(surf.mean(), 0.0)
        self.assertAlmostEqual(surf.rms_slope(), 0.0)
        self.assertAlmostEqual(surf.rms_curvature(), 0.0)

    def test_noniform_mean_zero(self):
        surface = fourier_synthesis((512, ), (1.3, ), 0.8, rms_height=1).to_nonuniform()
        self.assertTrue(not surface.is_uniform)
        x, h = surface.positions_and_heights()
        s, = surface.physical_sizes
        self.assertAlmostEqual(surface.mean(), np.trapz(h, x)/s)
        detrended_surface = surface.detrend(detrend_mode='height')
        self.assertAlmostEqual(detrended_surface.mean(), 0)
        x, h = detrended_surface.positions_and_heights()
        self.assertAlmostEqual(np.trapz(h, x), 0)

    def test_uniform_curvatures_2d(self):
        radius = 100.
        surface = make_sphere(radius, (160,131), (2., 3.),
                              kind="paraboloid")

        detrended = surface.detrend(detrend_mode="curvature")
        if False:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.contour(*surface.positions_and_heights())
            ax.set_aspect(1)
            plt.show(block=True)

        self.assertAlmostEqual(abs(detrended.curvatures[0]), 1 / radius)
        self.assertAlmostEqual(abs(detrended.curvatures[1]), 1 / radius)
        self.assertAlmostEqual(abs(detrended.curvatures[2]), 0)

    def test_uniform_curvatures_1d(self):
        radius = 100.
        surface = make_sphere(radius, (13, ), (6.,),
                              kind="paraboloid")

        detrended = surface.detrend(detrend_mode="curvature")
        if False:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.contour(*surface.positions_and_heights())
            ax.set_aspect(1)
            plt.show(block=True)

        self.assertAlmostEqual(abs(detrended.curvatures[0]), 1 / radius)

    def test_nonuniform_curvatures(self):
        radius = 400.
        center = 50.

        # generate a nonregular monotically increasing sery of x
        xs = np.cumsum(np.random.lognormal(size=20))
        xs /= np.max(xs)
        xs *= 80.

        heights = 1 / (2*radius) * (xs - center)**2

        surface = NonuniformLineScan(xs, heights)
        detrended = surface.detrend(detrend_mode="curvature")
        self.assertAlmostEqual(abs(detrended.curvatures[0]), 1 / radius)

class DetectFormatTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_detection(self):
        self.assertEqual(detect_format(os.path.join(DATADIR,  'di1.di')), 'di')
        self.assertEqual(detect_format(os.path.join(DATADIR,  'di2.di')), 'di')
        self.assertEqual(detect_format(os.path.join(DATADIR,  'example.ibw')), 'ibw')
        self.assertEqual(detect_format(os.path.join(DATADIR,  'example.opd')), 'opd')
        self.assertEqual(detect_format(os.path.join(DATADIR,  'example.x3p')), 'x3p')
        self.assertEqual(detect_format(os.path.join(DATADIR,  'example1.mat')), 'mat')
        self.assertEqual(detect_format(os.path.join(DATADIR,  'example.asc')), 'xyz')
        self.assertEqual(detect_format(os.path.join(DATADIR,  'line_scan_1_minimal_spaces.asc')), 'xyz')


class matSurfaceTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_read(self):
        from PyCo.Topography.IO import MatReader
        surface = MatReader(os.path.join(DATADIR,  'example1.mat')).topography(physical_sizes=[1.,1.])
        nx, ny = surface.nb_grid_pts
        self.assertEqual(nx, 2048)
        self.assertEqual(ny, 2048)
        self.assertAlmostEqual(surface.rms_height(), 1.234061e-07)
        self.assertTrue(surface.is_uniform)

    #TODO: test with multiple data

class npySurfaceTest(unittest.TestCase):
    def setUp(self):
        self.fn = "example.npy"
        self.res = (128,64)
        np.random.seed(1)
        self.data = np.random.random(self.res)
        self.data -= np.mean(self.data)

        np.save(self.fn, self.data)

    def test_read(self):
        size = (2,4)
        loader = NPYReader(self.fn)

        topo = loader.topography(physical_sizes=size)

        np.testing.assert_array_almost_equal(topo.heights(), self.data)

        #self.assertEqual(topo.info, loader.info)
        self.assertEqual(topo.physical_sizes, size)

    def tearDown(self):
        os.remove(self.fn)


class x3pSurfaceTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_read(self):
        surface = read_x3p(os.path.join(DATADIR,  'example.x3p'))
        nx, ny = surface.nb_grid_pts
        self.assertEqual(nx, 777)
        self.assertEqual(ny, 1035)
        sx, sy = surface.physical_sizes
        self.assertAlmostEqual(sx, 0.00068724)
        self.assertAlmostEqual(sy, 0.00051593)
        surface = read_x3p(os.path.join(DATADIR,  'example2.x3p'))
        nx, ny = surface.nb_grid_pts
        self.assertEqual(nx, 650)
        self.assertEqual(ny, 650)
        sx, sy = surface.physical_sizes
        self.assertAlmostEqual(sx, 8.29767313942749e-05)
        self.assertAlmostEqual(sy, 0.0002044783737930349)
        self.assertTrue(surface.is_uniform)

    def test_points_for_uniform_topography(self):
        surface = read_x3p(os.path.join(DATADIR,  'example.x3p'))
        x, y, z = surface.positions_and_heights()
        self.assertAlmostEqual(np.mean(np.diff(x[:, 0])), surface.physical_sizes[0] / surface.nb_grid_pts[0])
        self.assertAlmostEqual(np.mean(np.diff(y[0, :])), surface.physical_sizes[1] / surface.nb_grid_pts[1])


class opdSurfaceTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_read(self):
        surface = read_opd(os.path.join(DATADIR,  'example.opd'))
        nx, ny = surface.nb_grid_pts
        self.assertEqual(nx, 640)
        self.assertEqual(ny, 480)
        sx, sy = surface.physical_sizes
        self.assertAlmostEqual(sx, 0.125909140)
        self.assertAlmostEqual(sy, 0.094431855)
        self.assertTrue(surface.is_uniform)

    def test_undefined_points(self):
        t = read_opd(os.path.join(DATADIR, 'example2.opd'))
        self.assertTrue(t.has_undefined_data)


class diSurfaceTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_read(self):
        # All units are nm
        for (fn, n, s, rmslist) in [
            ('di1.di', 512, 500.0, [(9.9459868005603909,  "Height"),
                                         (114.01328027385664,  "Height"),
                                         (None, "Phase"),
                                         (None, "AmplitudeError")]),
            ('di2.di', 512, 300.0, [(24.721922008645919,"Height"),
                                         (24.807150576054838,"Height"),
                                         (0.13002312109876774, "Deflection")]),
            ('di3.di', 256, 10000.0, [(226.42539668457405, "ZSensor"),
                                           (None, "AmplitudeError"),
                                           (None, "Phase"),
                                           (264.00285276203158, "Height")]),  # Height
            ('di4.di', 512, 10000.0, [(81.622909804184744, "ZSensor"),  # ZSensor
                                           (0.83011806260022758, "AmplitudeError"),  # AmplitudeError
                                           (None, "Phase")])  # Phase
        ]:
            reader = open_topography(os.path.join(DATADIR, '{}').format(fn),
                                     format="di")


            for i, (rms, name) in enumerate(rmslist):
                assert reader.channels[i].name == name
                surface = reader.topography(channel_index=i)

                nx, ny = surface.nb_grid_pts
                self.assertEqual(nx, n)
                self.assertEqual(ny, n)
                sx, sy = surface.physical_sizes
                if type(surface.info['unit']) is tuple:
                    unit, dummy = surface.info['unit']
                else:
                    unit = surface.info['unit']
                self.assertAlmostEqual(sx * get_unit_conversion_factor(unit, 'nm'), s)
                self.assertAlmostEqual(sy * get_unit_conversion_factor(unit, 'nm'), s)
                if rms is not None:
                    self.assertAlmostEqual(surface.rms_height(), rms)
                    self.assertEqual(unit, 'nm')
                self.assertTrue(surface.is_uniform)

def test_di_orientation():
    #
    # topography.heights() should return an array where
    # - first index corresponds to x with lowest x in top rows and largest x in bottom rows
    # - second index corresponds to y with lowest y in left column and largest x in right column
    #
    # This is given when reading di.txt, see test elsewhere.
    #
    # The heights should be equal to those from txt reader
    di_fn = os.path.join(DATADIR, "di1.di")

    di_t = read_topography(di_fn)

    assert di_t.info['unit'] == 'nm'

    #
    # Check values in 4 corners, this should fix orientation
    #
    di_heights = di_t.heights()
    assert pytest.approx(di_heights[ 0,  0], abs=1e-3) == 6.060
    assert pytest.approx(di_heights[0, -1], abs=1e-3) == 2.843
    assert pytest.approx(di_heights[-1,  0], abs=1e-3) == -9.740
    assert pytest.approx(di_heights[-1, -1], abs=1e-3) == -30.306



class ibwSurfaceTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_read(self):
        reader = IBWReader(os.path.join(DATADIR, 'example.ibw'))
        surface = reader.topography()
        nx, ny = surface.nb_grid_pts
        self.assertEqual(nx, 512)
        self.assertEqual(ny, 512)
        sx, sy = surface.physical_sizes
        self.assertAlmostEqual(sx, 5.00978e-8)
        self.assertAlmostEqual(sy, 5.00978e-8)
        # self.assertEqual(surface.info['unit'], 'm')
        # Disabled unit check because I'm not sure
        # how to assign a valid unit to every channel - see IBW.py
        self.assertTrue(surface.is_uniform)

    def test_detect_format_then_read(self):
        f = open(os.path.join(DATADIR,  'example.ibw'), 'rb')
        fmt = detect_format(f)
        self.assertTrue(fmt, 'ibw')
        surface = open_topography(f, format=fmt).topography()
        f.close()


class hgtSurfaceTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_read(self):
        surface = read_hgt(os.path.join(DATADIR,  'N46E013.hgt'))
        nx, ny = surface.nb_grid_pts
        self.assertEqual(nx, 3601)
        self.assertEqual(ny, 3601)
        self.assertTrue(surface.is_uniform)


class h5SurfaceTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_detect_format(self):

        self.assertEqual(PyCo.Topography.IO.detect_format( # TODO: this will be the standart detect format method in the future
            os.path.join(DATADIR,  'surface.2048x2048.h5')), 'h5')

    def test_read(self):
        loader = H5Reader(os.path.join(DATADIR,  'surface.2048x2048.h5'))

        topography = loader.topography(physical_sizes=(1.,1.))
        nx, ny = topography.nb_grid_pts
        self.assertEqual(nx, 2048)
        self.assertEqual(ny, 2048)
        self.assertTrue(topography.is_uniform)
        self.assertEqual(topography.dim, 2)


class xyzSurfaceTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_detect_format_then_read(self):
        self.assertEqual(detect_format(os.path.join(DATADIR,  'example.asc')), 'xyz')

    def test_read(self):
        surface = read_xyz(os.path.join(DATADIR,  'example.asc'))
        self.assertFalse(surface.is_uniform)
        x, y = surface.positions_and_heights()
        self.assertGreater(len(x), 0)
        self.assertEqual(len(x), len(y))
        self.assertFalse(surface.is_uniform)
        self.assertEqual(surface.dim, 1)
        self.assertFalse(surface.is_periodic)


class PipelineTests(unittest.TestCase):
    def test_scaled_topography(self):
        surf = read_xyz(os.path.join(DATADIR,  'example.asc'))
        for fac in [1.0, 2.0, np.pi]:
            surf2 = surf.scale(fac)
            self.assertAlmostEqual(fac * surf.rms_height(kind='Rq'), surf2.rms_height(kind='Rq'))

    def test_transposed_topography(self):
        surf = fourier_synthesis([124, 368], [6, 3], 0.8, rms_slope=0.1)
        nx, ny = surf.nb_grid_pts
        sx, sy = surf.physical_sizes
        surf2 = surf.transpose()
        nx2, ny2 = surf2.nb_grid_pts
        sx2, sy2 = surf2.physical_sizes
        self.assertEqual(nx, ny2)
        self.assertEqual(ny, nx2)
        self.assertEqual(sx, sy2)
        self.assertEqual(sy, sx2)
        self.assertTrue((surf.heights() == surf2.heights().T).all())

class ScalarParametersTest(PyCoTestCase):
    @unittest.skip
    def test_rms_slope_1d(self):
        r = 4096
        res = (r, )
        for H in [0.3, 0.8]:
            for s in [(1, ), (1.4, )]:
                t = fourier_synthesis(res, s, H, short_cutoff=32 / r * np.mean(s), rms_slope=0.1,
                                      amplitude_distribution=lambda n: 1.0)
                self.assertAlmostEqual(t.rms_slope(), 0.1, places=2)

    def test_rms_slope_2d(self):
        r = 2048
        res = [r, r]
        for H in [0.3, 0.8]:
            for s in [(1, 1), (1.4, 3.3)]:
                t = fourier_synthesis(res, s, H, short_cutoff=8 / r * np.mean(s), rms_slope=0.1,
                                      amplitude_distribution=lambda n: 1.0)
                self.assertAlmostEqual(t.rms_slope(), 0.1, places=2)


class ConvertersTest(PyCoTestCase):
    def test_wrapped_x_range(self):
        t = fourier_synthesis((128, ), (1, ), 0.8, rms_slope=0.1).to_nonuniform()
        x = t.positions()
        self.assertAlmostEqual(t.x_range[0], x[0])
        self.assertAlmostEqual(t.x_range[1], x[-1])

    def test_delegation(self):
        t1 = fourier_synthesis((128, ), (1, ), 0.8, rms_slope=0.1)
        t2 = t1.detrend()
        t3 = t2.to_nonuniform()
        t4 = t3.scale(2.0)

        t4.scale_factor
        with self.assertRaises(AttributeError):
            t2.scale_factor
        t2.detrend_mode
        # detrend_mode should not be delegated to the parent class
        with self.assertRaises(AttributeError):
            t4.detrend_mode

        # t2 should have 'to_nonuniform'
        t2.to_nonuniform()
        # but it should not have 'to_uniform'
        with self.assertRaises(AttributeError):
            t2.to_uniform(100, 10)

        # t4 should have 'to_uniform'
        t5 = t4.to_uniform(100, 10)
        # but it should not have 'to_nonuniform'
        with self.assertRaises(AttributeError):
            t4.to_nonuniform()

        # t5 should have 'to_nonuniform'
        t5.to_nonuniform()
        # but it should not have 'to_uniform'
        with self.assertRaises(AttributeError):
            t5.to_uniform(100, 10)

    def test_autocompletion(self):
        t1 = fourier_synthesis((128, ), (1, ), 0.8, rms_slope=0.1)
        t2 = t1.detrend()
        t3 = t2.to_nonuniform()

        self.assertIn('detrend', dir(t1))
        self.assertIn('to_nonuniform', dir(t2))
        self.assertIn('to_uniform', dir(t3))


class DerivativeTest(PyCoTestCase):
    def test_uniform_vs_nonuniform(self):
        t1 = fourier_synthesis([12], [6], 0.8, rms_slope=0.1)
        t2 = t1.to_nonuniform()

        d1 = t1.derivative(1)
        d2 = t2.derivative(1)

        self.assertArrayAlmostEqual(d1[:-1], d2)

    def test_analytic(self):
        nb_pts = 1488
        s = 4*np.pi
        x = np.arange(nb_pts)*s/nb_pts
        h = np.sin(x)
        t1 = UniformLineScan(h, (s,))
        t2 = UniformLineScan(h, (s,), periodic=True)
        t3 = t1.to_nonuniform()

        d1 = t1.derivative(1)
        d2 = t2.derivative(1)
        d3 = t3.derivative(1)

        self.assertArrayAlmostEqual(d1, np.cos(x[:-1]+(x[1]-x[0])/2), tol=1e-5)
        self.assertArrayAlmostEqual(d2, np.cos(x+(x[1]-x[0])/2), tol=1e-5)
        self.assertArrayAlmostEqual(d3, np.cos(x[:-1]+(x[1]-x[0])/2), tol=1e-5)

        d1 = t1.derivative(2)
        d2 = t2.derivative(2)
        d3 = t3.derivative(2)

        self.assertArrayAlmostEqual(d1, -np.sin(x[:-2]+(x[1]-x[0])), tol=1e-5)
        self.assertArrayAlmostEqual(d2, -np.sin(x+(x[1]-x[0])), tol=1e-5)
        self.assertArrayAlmostEqual(d3, -np.sin(x[:-2]+(x[1]-x[0])), tol=1e-5)


class PickeTest(PyCoTestCase):
    def test_detrended(self):
        t1 = read_topography(os.path.join(DATADIR, 'di1.di'))

        dt1 = t1.detrend(detrend_mode="center")

        t1_pickled = pickle.dumps(t1)
        t1_unpickled = pickle.loads(t1_pickled)

        dt2 = t1_unpickled.detrend(detrend_mode="center")

        self.assertAlmostEqual(dt1.coeffs[0], dt2.coeffs[0])


def test_txt_example():
    t = read_topography(os.path.join(DATADIR, 'txt_example.txt'))
    assert t.physical_sizes == (1e-6, 0.5e-6)
    assert t.nb_grid_pts == (6, 3)

    expected_heights = np.array([
        [1.0e-007, 0.0e-007, 0.0e-007, 0.0e-007, 0.0e-007, 0.0e-007],
        [0.5e-007, 0.5e-007, 0.0e-008, 0.0e-008, 0.0e-008, 0.0e-008],
        [0.0e-007, 0.0e-007, 0.0e-007, 0.0e-007, 0.0e-007, 0.0e-007],
    ]).T

    np.testing.assert_allclose(t.heights(), expected_heights)
