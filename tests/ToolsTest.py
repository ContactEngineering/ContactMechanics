#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   ToolsTest.py

@author Till Junge <till.junge@kit.edu>

@date   13 Feb 2015

@brief  Tests for PyCo helper tools

@section LICENCE

Copyright 2015-2017 Till Junge, Lars Pastewka

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

try:
    import unittest
    import numpy as np
    import warnings

    import PyCo.Topography.Nonuniform as Nonuniform
    import PyCo.Topography.Uniform as Uniform
    from PyCo.Tools import evaluate_gradient, mean_err
    from PyCo.Topography import (autocorrelation_1D, autocorrelation_2D, tilt_from_height,
                                 shift_and_tilt, shift_and_tilt_approx, shift_and_tilt_from_slope,
                                 NonuniformNumpyTopography, UniformNumpyTopography)
    from PyCo.Topography.Generation import RandomSurfaceGaussian, RandomSurfaceExact

    from .PyCoTest import PyCoTestCase
except ImportError as err:
    import sys
    print(err)
    sys.exit(-1)

class ToolTest(PyCoTestCase):
    def test_gradient(self):
        coeffs = np.random.random(2)+1.
        fun = lambda x: (coeffs*x**2).sum()
        grad= lambda x: 2*coeffs*x

        x= 20*(np.random.random(2)-.5)
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
        arr = np.arange(5)*a+d
        arr_out = shift_and_tilt(arr)
        self.assertTrue(arr_out.sum() <tol, "{}".format(arr_out))

        # 2D
        arr = arr + np.arange(6).reshape((-1, 1))*b
        arr_out = shift_and_tilt(arr)
        error = arr_out.sum()
        self.assertTrue(error <tol, "error = {}, tol = {}, arr_out = {}".format(
            error, tol, arr_out))

        self.assertTrue(arr.shape == arr_out.shape,
                    "arr.shape = {}, arr_out.shape = {}".format(
                        arr.shape, arr_out.shape))

        arr_approx, x = shift_and_tilt_approx(arr, full_output=True)
        error  =arr_approx.sum()
        self.assertTrue(error < tol, "error = {}, tol = {}, arr_out = {}".format(
            error, tol, arr_approx))

        mean_slope = [np.diff(arr, axis=d).mean() for d in range(len(arr.shape))]
        arr_out = shift_and_tilt_from_slope(arr)
        mean_slope = [np.diff(arr_out, axis=d).mean() for d in range(len(arr_out.shape))]
        self.assertAlmostEqual(mean_slope[0], 0)
        self.assertAlmostEqual(mean_slope[1], 0)

        mean_slope = tilt_from_height(arr)
        self.assertAlmostEqual(mean_slope[0], b)
        self.assertAlmostEqual(mean_slope[1], a)
        self.assertAlmostEqual(mean_slope[2], d)

        mean_slope = tilt_from_height(UniformNumpyTopography(arr))
        self.assertAlmostEqual(mean_slope[0], b)
        self.assertAlmostEqual(mean_slope[1], a)
        self.assertAlmostEqual(mean_slope[2], d)

        mean_slope = tilt_from_height(UniformNumpyTopography(arr, size=(3, 2.5)))
        self.assertAlmostEqual(mean_slope[0], 2*b)
        self.assertAlmostEqual(mean_slope[1], 2*a)
        self.assertAlmostEqual(mean_slope[2], d)


    def test_impulse_autocorrelation(self):
        nx = 16
        for x, w, h, p in [(nx//2, 3, 1, True), (nx//3, 2, 2, True),
                           (nx//2, 5, 1, False), (nx//3, 6, 2.5, False)]:
            y = np.zeros([nx, 1])
            y[x-w//2:x+(w+1)//2] = h
            r, A = autocorrelation_1D(y, periodic=True)

            A_ana = np.zeros_like(A)
            A_ana[:w] = h**2*np.linspace(w/nx, 1/nx, w)
            A_ana = A_ana[0] - A_ana
            self.assertTrue(np.allclose(A, A_ana))


    def test_brute_force_autocorrelation_1D(self):
        n = 10
        for surf in [UniformNumpyTopography(np.ones(n).reshape(n, 1)),
                     UniformNumpyTopography(np.arange(n).reshape(n, 1)),
                     UniformNumpyTopography(np.random.random(n).reshape(n, 1))]:
            r, A = autocorrelation_1D(surf, periodic=False)

            n = len(A)
            dir_A = np.zeros(n)
            for d in range(n):
                for i in range(n-d):
                    dir_A[d] += (surf[i] - surf[i+d])**2/2
                dir_A[d] /= (n-d)
            self.assertArrayAlmostEqual(A, dir_A)


    def test_brute_force_autocorrelation_2D(self):
        n = 10
        m = 11
        for surf in [UniformNumpyTopography(np.ones([n, m])),
                     UniformNumpyTopography(np.random.random([n, m]))]:
            r, A, A_xy = autocorrelation_2D(surf, periodic=False, return_map=True)

            nx, ny = surf.shape
            dir_A_xy = np.zeros([n, m])
            dir_A = np.zeros_like(A)
            dir_n = np.zeros_like(A)
            for dx in range(n):
                for dy in range(m):
                    for i in range(nx-dx):
                        for j in range(ny-dy):
                            dir_A_xy[dx, dy] += (surf[i, j] - surf[i+dx, j+dy])**2/2
                    dir_A_xy[dx, dy] /= (nx-dx)*(ny-dy)
                    d = np.sqrt(dx**2 + dy**2)
                    i = np.argmin(np.abs(r-d))
                    dir_A[i] += dir_A_xy[dx, dy]
                    dir_n[i] += 1
            dir_n[dir_n==0] = 1
            dir_A /= dir_n
            self.assertArrayAlmostEqual(A_xy, dir_A_xy)
            self.assertArrayAlmostEqual(A[:-2], dir_A[:-2])


    def test_nonuniform_rms_height(self):
        n = 1024
        dx = 0.12
        # make a function that is smooth on short scales
        h = np.fft.irfft(np.exp(1j*np.random.random(n//2+1))*(np.arange(n//2+1) < (n//64)))
        h1 = Uniform.rms_height(h)
        x = np.arange(n)*dx
        h2 = Nonuniform.rms_height(x, h)
        self.assertAlmostEqual(h1, h2, places=3)