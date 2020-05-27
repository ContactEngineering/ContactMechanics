#
# Copyright 2018-2019 Antoine Sanner
#           2018-2019 Lars Pastewka
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
Tests for PyCo Goodies
"""


import unittest
import numpy as np
import warnings

import PyCo.Goodies as Goodies
from PyCo.Topography.Generation import fourier_synthesis, self_affine_prefactor

from NuMPI import MPI
import pytest
pytestmark = pytest.mark.skipif(MPI.COMM_WORLD.Get_size()> 1,
        reason="tests only serial funcionalities, please execute with pytest")


class GoodiesTest(unittest.TestCase):
    def test_surf_param_recovery(self):
        warnings.filterwarnings("error")
        siz = 3
        size = (siz, siz)
        hurst = .9
        rms_height = 1
        res = 101 # FIXME: BUG: now it suddenly doesn't reach the same tolerance for 100 points, I don't understand why
        nb_grid_pts = (res, res)
        lam_max = .5
        lam_min = 2  / np.min(np.asarray(nb_grid_pts) / np.asarray(size))
        surf = fourier_synthesis(nb_grid_pts, size, hurst, rms_height,
                                     long_cutoff=lam_max,
                                     short_cutoff=lam_min,
                                     rolloff=0,
                                     amplitude_distribution=lambda n: np.ones(n))
        prefactor_in = (self_affine_prefactor(nb_grid_pts, size, hurst, rms_height,
                                             long_cutoff=lam_max, short_cutoff=lam_min)
                            / np.prod(nb_grid_pts) * np.sqrt(np.prod(size)))**2


        rms_height_fromC_in = surf.rms_height()

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

        # (surf_gen.compute_prefactor()/np.sqrt(np.prod(size)))**2
        error = abs(1-prefactor_in/prefactor_out)
        self.assertTrue(error < reproduction_tol,
                        "Error = {}, β_in = {}, β_out = {}".format(
                            error, prefactor_in, prefactor_out))

    def test_surf_param_recovery_weighted(self):
        siz = 3
        size = (siz, siz)
        hurst = .9
        rms_height = 1
        res = 101 # FIXME: BUG: now it suddenly doesn't reach the same tolerance for 100 points, I don't understand why
        nb_grid_pts = (res, res)
        lam_max = .5
        lam_min = 2  / np.min(np.asarray(nb_grid_pts) / np.asarray(size))
        surf = fourier_synthesis(nb_grid_pts, size, hurst, rms_height,
                                     long_cutoff=lam_max,
                                     short_cutoff=lam_min,
                                     rolloff=0,
                                     amplitude_distribution=lambda n: np.ones(n))
        prefactor_in = (self_affine_prefactor(nb_grid_pts, size, hurst, rms_height,
                                             long_cutoff=lam_max, short_cutoff=lam_min)
                            / np.prod(nb_grid_pts) * np.sqrt(np.prod(size)))**2
        surf_char = Goodies.CharacterisePeriodicSurface(surf)
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
        nb_grid_pts = (res, res)
        lam_max = .5
        lam_min = 2  / np.min(np.asarray(nb_grid_pts) / np.asarray(size))
        np.random.seed(10)
        surf = fourier_synthesis(nb_grid_pts, size, hurst, rms_height,
                                     long_cutoff=lam_max,
                                     short_cutoff=lam_min,
                                     rolloff=0)
        prefactor_in = (self_affine_prefactor(nb_grid_pts, size, hurst, rms_height,
                                             long_cutoff=lam_max, short_cutoff=lam_min)
                            / np.prod(nb_grid_pts) * np.sqrt(np.prod(size)))**2
        surf_char = Goodies.CharacterisePeriodicSurface(surf)
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
