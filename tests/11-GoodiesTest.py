#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   11-GoodiesTest.py

@author Till Junge <till.junge@kit.edu>

@date   02 Feb 2016

@brief  Tests for PyCo Goodies

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

    import PyCo.Goodies as Goodies
except ImportError as err:
    import sys
    print(err)
    sys.exit(-1)

class GoodiesTest(unittest.TestCase):
    def test_surf_param_recovery(self):
        warnings.filterwarnings("error")
        siz = 3
        size = (siz, siz)
        hurst = .9
        rms_height = 1
        res = 100
        resolution = (res, res)
        lam_max = .5
        surf_gen = Goodies.RandomSurfaceExact(resolution, size, hurst,
                                            rms_height, lambda_max=lam_max)
        surf = surf_gen.get_surface(roll_off=0, lambda_max=lam_max)
        rms_height_fromC_in = surf.rms_height_q_space()

        error = abs(1-rms_height_fromC_in/rms_height)
        rough_tol = .02
        self.assertTrue(error < rough_tol,
                        "Error = {}, rms_height_in = {}, rms_height_out = {}".format(
                            error, rms_height_fromC_in, rms_height))

        surf_char = Goodies.CharacterisePeriodicSurface(surf)
        rms_height_out = surf_char.compute_rms_height()
        reproduction_tol = 1e-5
        error = abs(1 - rms_height_out/rms_height_fromC_in)
        self.assertTrue(error < reproduction_tol)

        hurst_out, prefactor_out = surf_char.estimate_hurst(
            full_output=True, lambda_max=lam_max)

        error = abs(1-hurst/hurst_out)
        self.assertTrue(error < reproduction_tol, "Error = {}, reproduction_tol = {}, hurst = {}, hurst_out= {}".format(error, reproduction_tol, hurst, hurst_out))

        prefactor_in = (surf_gen.compute_prefactor()/np.sqrt(np.prod(size)))**2
        error = abs(1-prefactor_in/prefactor_out)
        self.assertTrue(error < reproduction_tol,
                        "Error = {}, β_in = {}, β_out = {}".format(
                            error, prefactor_in, prefactor_out))

    def test_surf_param_recovery_weighted(self):
        siz = 3
        size = (siz, siz)
        hurst = .9
        rms_height = 1
        res = 100
        resolution = (res, res)
        lam_max = .5
        surf_gen = Goodies.RandomSurfaceExact(resolution, size, hurst,
                                            rms_height, lambda_max=lam_max)
        surf = surf_gen.get_surface(roll_off=0, lambda_max=lam_max)
        surf_char = Goodies.CharacterisePeriodicSurface(surf)
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


    def test_surf_param_recovery_weighted_gaussian(self):
        siz = 3
        size = (siz, siz)
        hurst = .9
        rms_height = 1
        res = 100
        resolution = (res, res)
        lam_max = .5
        surf_gen = Goodies.RandomSurfaceGaussian(resolution, size, hurst,
                                               rms_height, lambda_max=lam_max,
                                               seed=10)
        surf = surf_gen.get_surface(roll_off=0, lambda_max=lam_max)
        surf_char = Goodies.CharacterisePeriodicSurface(surf)
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
