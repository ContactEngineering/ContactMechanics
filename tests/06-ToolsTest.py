#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   06-ToolsTest.py

@author Till Junge <till.junge@kit.edu>

@date   13 Feb 2015

@brief  Tests for PyPyContact helper tools

@section LICENCE

 Copyright (C) 2015 Till Junge

PyPyContact is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation, either version 3, or (at
your option) any later version.

PyPyContact is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with GNU Emacs; see the file COPYING. If not, write to the
Free Software Foundation, Inc., 59 Temple Place - Suite 330,
Boston, MA 02111-1307, USA.
"""

try:
    import unittest
    import numpy as np

    import PyPyContact.Tools as Tools
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
        approx_g = Tools.evaluate_gradient(fun, x, 1e-5)
        error = Tools.mean_err(g, approx_g)

        msg = []
        msg.append("f = {}".format(f))
        msg.append("g = {}".format(g))
        msg.append('approx = {}'.format(approx_g))
        msg.append("error = {}".format(error))
        msg.append("tol = {}".format(tol))
        self.assertTrue(error < tol, ", ".join(msg))


    def test_surf_param_recovery(self):
        siz = 3
        size = (siz, siz)
        hurst = .9
        h_rms = 1
        res = 100
        resolution = (res, res)
        lam_max = .5
        surf_gen = Tools.RandomSurfaceExact(resolution, size, hurst,
                                            h_rms, lambda_max=lam_max)
        surf = surf_gen.get_surface(roll_off=0, lambda_max=lam_max)
        h_rms_fromC_in = surf.compute_h_rms_fromReciprocSpace()

        error = abs(1-h_rms_fromC_in/h_rms)
        rough_tol = .02
        self.assertTrue(error < rough_tol,
                        "Error = {}, h_rms_in = {}, h_rms_out = {}".format(
                            error, h_rms_fromC_in, h_rms))

        surf_char = Tools.CharacterisePeriodicSurface(surf)
        h_rms_out = surf_char.compute_h_rms()
        reproduction_tol = 1e-5
        error = abs(1 - h_rms_out/h_rms_fromC_in)
        self.assertTrue(error < reproduction_tol)

        hurst_out, prefactor_out = surf_char.estimate_hurst(
            full_output=True, lambda_max=lam_max)

        error = abs(1-hurst/hurst_out)
        self.assertTrue(error < reproduction_tol)

        prefactor_in = (surf_gen.compute_prefactor()/np.sqrt(np.prod(size)))**2
        error = abs(1-prefactor_in/prefactor_out)
        self.assertTrue(error < reproduction_tol,
                        "Error = {}, β_in = {}, β_out = {}".format(
                            error, prefactor_in, prefactor_out))

    def test_surf_param_recovery_weighted(self):
        siz = 3
        size = (siz, siz)
        hurst = .9
        h_rms = 1
        res = 100
        resolution = (res, res)
        lam_max = .5
        surf_gen = Tools.RandomSurfaceExact(resolution, size, hurst,
                                            h_rms, lambda_max=lam_max)
        surf = surf_gen.get_surface(roll_off=0, lambda_max=lam_max)
        surf_char = Tools.CharacterisePeriodicSurface(surf)
        prefactor_in = (surf_gen.compute_prefactor()/np.sqrt(np.prod(size)))**2
        hurst_out, prefactor_out = surf_char.estimate_hurst(
            lambda_max=lam_max, full_output=True)
        hurst_error = abs(1-hurst_out/hurst)
        prefactor_error = abs(1-prefactor_out/prefactor_in)
        reproduction_tol = 1e-5
        self.assertTrue(hurst_error<reproduction_tol,
                        "error = {}, h_out = {}, h_in = {}, tol = {}".format(
                            hurst_error, hurst_out, hurst, reproduction_tol))
        self.assertTrue(
            prefactor_error<reproduction_tol,
            "C0_err = {}, tol = {}".format(prefactor_error, reproduction_tol))

    def test_compare_exact_1Dvs2D_power_spectrum(self):
        siz = 3
        size = (siz, siz)
        hurst = .9
        h_rms = 1
        res = 100
        resolution = (res, res)
        lam_max = .5
        surf_gen = Tools.RandomSurfaceExact(resolution, size, hurst,
                                            h_rms, lambda_max=lam_max)
        surf = surf_gen.get_surface(roll_off=0, lambda_max=lam_max)
        surf_char2D = Tools.CharacterisePeriodicSurface(surf)
        surf_char1D = Tools.CharacterisePeriodicSurface(surf, one_dimensional=True)

        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(surf_char1D.q, surf_char1D.C, label="1D")
        ax.plot(surf_char2D.q, surf_char2D.C, label="2D", ls='--')
        ax.legend(loc='best')
        plt.show()
        hurst_out2D, prefactor_out2D = surf_char2D.estimate_hurst(full_output=True)
        hurst_out1D, prefactor_out1D = surf_char1D.estimate_hurst(full_output=True)

        self.assertTrue(hurst_out1D == hurst_out2D, "1D: {},\n2D{}".format(hurst_out1D, hurst_out2D))


    def test_surf_param_recovery_weighted_gaussian(self):
        siz = 3
        size = (siz, siz)
        hurst = .9
        h_rms = 1
        res = 100
        resolution = (res, res)
        lam_max = .5
        surf_gen = Tools.RandomSurfaceGaussian(resolution, size, hurst,
                                               h_rms, lambda_max=lam_max,
                                               seed=10)
        surf = surf_gen.get_surface(roll_off=0, lambda_max=lam_max)
        surf_char = Tools.CharacterisePeriodicSurface(surf)
        prefactor_in = (surf_gen.compute_prefactor()/np.sqrt(np.prod(size)))**2
        hurst_out, prefactor_out = surf_char.estimate_hurst(
            lambda_max=lam_max, full_output=True)
        hurst_error = abs(1-hurst_out/hurst)
        prefactor_error = abs(1-prefactor_out/prefactor_in)
        reproduction_tol = .03
        self.assertTrue(
            hurst_error<reproduction_tol,
            "error = {}, h_out = {}, h_in = {}, tol = {}".format(
                hurst_error, hurst_out, hurst, reproduction_tol))
        self.assertTrue(prefactor_error < 1,
            "error = {}, C0_out = {}, C0_in = {}, tol = {}".format(
                prefactor_error, prefactor_out, prefactor_in, reproduction_tol))


    def test_shift_and_tilt(self):
        tol = 1e-10
        a = 1.2
        b = 2.5
        d = .2
        # 1D
        arr = np.arange(5)*a+d
        arr_out = Tools.shift_and_tilt(arr)
        self.assertTrue(arr_out.sum() <tol, "{}".format(arr_out))

        # 2D
        arr = arr + np.arange(6).reshape((-1, 1))*b
        arr_out = Tools.shift_and_tilt(arr)
        error = arr_out.sum()
        self.assertTrue(error <tol, "error = {}, tol = {}, arr_out = {}".format(
            error, tol, arr_out))

        self.assertTrue(arr.shape == arr_out.shape,
                    "arr.shape = {}, arr_out.shape = {}".format(
                        arr.shape, arr_out.shape))

        arr_approx, x = Tools.shift_and_tilt_approx(arr, full_output=True)
        error  =arr_approx.sum()
        self.assertTrue(error < tol, "error = {}, tol = {}, arr_out = {}".format(
            error, tol, arr_approx))
