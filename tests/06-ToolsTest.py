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

