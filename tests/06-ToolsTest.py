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

    def test_random_surface(self):
        tol = 1e-8
        resolution = (5, 5)
        size = 6.
        hurst = .8
        h_rms = 2
        rs = Tools.RandomSurfaceExact(resolution, size, hurst, h_rms)
        error = Tools.mean_err(np.fft.ifftn(rs.coeffs),
                               rs.get_surface().profile())
        self.assertTrue(error < tol)

        rsGauss = Tools.RandomSurfaceGaussian(resolution, size, hurst, h_rms)
        error = Tools.mean_err(np.fft.ifftn(rsGauss.coeffs*rsGauss.distribution),
                               rsGauss.get_surface().profile())
        msg = "error = {}, computed:\n{}\nfake\n{}:\noutput:\n{}".format(
            error, np.fft.ifftn(rsGauss.coeffs*rsGauss.distribution),
            np.fft.ifftn(rsGauss.coeffs),
            rsGauss.get_surface())
        self.assertTrue(error < tol, msg)


    def test_surf_analysis(self):
        resolution = (1000, 1000)
        size = 12.
        hurst = .8
        h_rms = 2
        rs = Tools.RandomSurfaceGaussian(resolution, size, hurst, h_rms)
        surf_char = Tools.CharacterisePeriodicSurface(rs.get_surface(lambda_max=(2*np.pi/10),lambda_min=(2*np.pi/140)))
        import matplotlib.pyplot as plt
        q = surf_char.q
        C = surf_char.C
        fig = plt.figure()
        ax=fig.add_subplot(111)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_ylim(bottom=1e-16)
        plt.loglog(q, C, alpha=.1)
        mean, err, q_g = surf_char.grouped_stats(100)
        ax.errorbar(q_g, mean, yerr=err)
        print(rs.get_surface().profile().mean())
        print("(min, max)(C) : {}".format((C.min(), C.max())))
        plt.show()
