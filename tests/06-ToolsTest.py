#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   06-ToolsTest.py

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

    from PyCo.Tools import evaluate_gradient, mean_err
    from PyCo.Topography import (autocorrelation_1D, compute_derivative, tilt_from_height, shift_and_tilt,
                                 shift_and_tilt_approx, shift_and_tilt_from_slope, NumpyTopography)
    from PyCo.Goodies import RandomSurfaceGaussian
except ImportError as err:
    import sys
    print(err)
    sys.exit(-1)

class ToolTest(unittest.TestCase):
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


##     def test_compare_exact_1Dvs2D_power_spectrum(self):
##         siz = 3
##         size = (siz, siz)
##         hurst = .9
##         rms_height = 1
##         res = 100
##         resolution = (res, res)
##         lam_max = .5
##         surf_gen = Tools.RandomSurfaceExact(resolution, size, hurst,
##                                             rms_height, lambda_max=lam_max)
##         surf = surf_gen.get_surface(roll_off=0, lambda_max=lam_max)
##         surf_char2D = Tools.CharacterisePeriodicSurface(surf)
##         surf_char1D = Tools.CharacterisePeriodicSurface(surf, one_dimensional=True)
## 
##         import matplotlib.pyplot as plt
##         fig = plt.figure()
##         ax = fig.add_subplot(111)
##         ax.plot(surf_char1D.q, surf_char1D.C, label="1D")
##         ax.plot(surf_char2D.q, surf_char2D.C, label="2D", ls='--')
##         ax.legend(loc='best')
##         plt.show()
##         hurst_out2D, prefactor_out2D = surf_char2D.estimate_hurst(full_output=True)
##         hurst_out1D, prefactor_out1D = surf_char1D.estimate_hurst(full_output=True)
## 
##         self.assertTrue(hurst_out1D == hurst_out2D, "1D: {},\n2D{}".format(hurst_out1D, hurst_out2D))



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

        mean_slope = [x.mean() for x in compute_derivative(arr)]
        arr_out = shift_and_tilt_from_slope(arr)
        mean_slope = [x.mean() for x in compute_derivative(arr_out)]
        self.assertAlmostEqual(mean_slope[0], 0)
        self.assertAlmostEqual(mean_slope[1], 0)

        mean_slope = tilt_from_height(arr)
        self.assertAlmostEqual(mean_slope[0], b)
        self.assertAlmostEqual(mean_slope[1], a)
        self.assertAlmostEqual(mean_slope[2], d)

        mean_slope = tilt_from_height(NumpyTopography(arr))
        self.assertAlmostEqual(mean_slope[0], b)
        self.assertAlmostEqual(mean_slope[1], a)
        self.assertAlmostEqual(mean_slope[2], d)

        mean_slope = tilt_from_height(NumpyTopography(arr, size=(3, 2.5)))
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