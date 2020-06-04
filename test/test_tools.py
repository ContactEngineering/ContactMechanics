#
# Warning: Could no find author name for junge@cmsjunge
# Copyright 2016, 2018, 2020 Lars Pastewka
#           2018, 2020 Antoine Sanner
#           2015-2016 Till Junge
#           2015 junge@cmsjunge
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
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
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
Tests for PyCo helper tools
"""

import unittest

import numpy as np

from ContactMechanics.Tools import evaluate_gradient, mean_err
from SurfaceTopography import Topography, UniformLineScan, NonuniformLineScan
from SurfaceTopography.Nonuniform.Detrending import polyfit
from SurfaceTopography.Uniform.Detrending import tilt_from_height, \
    shift_and_tilt

import pytest
from NuMPI import MPI

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="test only serial functionalities, please execute with pytest")


class ToolTest(unittest.TestCase):
    def test_gradient(self):
        coeffs = np.random.random(2) + 1.

        def fun(x):
            return (coeffs * x ** 2).sum()

        def grad(x):
            return 2 * coeffs * x

        x = 20 * (np.random.random(2) - .5)
        tol = 1e-8
        f = fun(x)
        g = grad(x)
        approx_g = evaluate_gradient(fun, x, 1e-5)
        error = mean_err(g, approx_g)

        msg = []
        msg.append("f = {}".format(f))
        msg.append("g = {}".format(g))
        msg.append('approx = {}'.format(approx_g))
        msg.append("error = {}".format(error))
        msg.append("tol = {}".format(tol))
        self.assertTrue(error < tol, ", ".join(msg))

    def test_shift_and_tilt(self):
        tol = 1e-10
        a = 1.2
        b = 2.5
        d = .2
        # 1D
        arr = np.arange(5) * a + d
        arr_out = shift_and_tilt(UniformLineScan(arr, arr.shape))
        self.assertTrue(arr_out.sum() < tol, "{}".format(arr_out))

        # 2D
        arr = arr + np.arange(6).reshape((-1, 1)) * b
        arr_out = shift_and_tilt(Topography(arr, arr.shape))
        error = arr_out.sum()
        self.assertTrue(error < tol,
                        "error = {}, tol = {}, arr_out = {}".format(
                            error, tol, arr_out))

        self.assertTrue(arr.shape == arr_out.shape,
                        "arr.shape = {}, arr_out.shape = {}".format(
                            arr.shape, arr_out.shape))

        nx, ny = arr.shape
        mean_slope = tilt_from_height(Topography(arr, arr.shape))
        np.testing.assert_allclose(mean_slope[0], b * nx)
        np.testing.assert_allclose(mean_slope[1], a * ny)
        np.testing.assert_allclose(mean_slope[2], d)

    def test_nonuniform_rms_height(self):
        n = 1024
        dx = 0.12
        # make a function that is smooth on short scales
        h = np.fft.irfft(np.exp(1j * np.random.random(n // 2 + 1)) * (
                np.arange(n // 2 + 1) < (n // 64)))
        h1 = UniformLineScan(h, h.shape).rms_height()
        x = np.arange(n) * dx
        h2 = NonuniformLineScan(x, h).rms_height()
        np.testing.assert_allclose(h1, h2, rtol=1.5e-2)

    def test_polynomial_fit(self):
        x = np.linspace(0, 10, 11) ** 2
        y = 1.8 * x + 1.2
        b, m = polyfit(x, y, 1)
        np.testing.assert_allclose(b, 1.2)
        np.testing.assert_allclose(m, 1.8)

        y += 0.1 * np.sin(2 * np.pi * x)
        b, m = polyfit(x, y, 1)
        np.testing.assert_allclose(b, 1.2)
        np.testing.assert_allclose(m, 1.8)

        b, m, m2 = polyfit(x, y, 2)
        np.testing.assert_allclose(b, 1.2)
        np.testing.assert_allclose(m, 1.8)
        np.testing.assert_allclose(m2, 0.0, atol=1e-7)

        y = 1.8 * x + 1.2 + 2.3 * x ** 2
        b, m, m2 = polyfit(x, y, 2)
        np.testing.assert_allclose(b, 1.2)
        np.testing.assert_allclose(m, 1.8)
        np.testing.assert_allclose(m2, 2.3)
