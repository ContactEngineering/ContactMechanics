#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   12-ReferenceSolutionsTest.py

@author Lars Pastewka <lars.pastewka@kit.edu>

@date   24 April 2016

@brief  Tests for PyCo ReferenceSolutions

@section LICENCE

 Copyright (C) 2016 Lars Pastewka

PyCo is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation, either version 3, or (at
your option) any later version.

PyCo is distributed in the hope that it will be useful, but
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
    import warnings
    from scipy.integrate import quad
    from scipy.special import ellipk, erf

    from .PyCoTest import PyCoTestCase
    import PyCo.ReferenceSolutions.GreenwoodTripp as GT
    import PyCo.ReferenceSolutions.MaugisDugdale as MD
except ImportError as err:
    import sys
    print(err)
    sys.exit(-1)

class ReferenceSolutionsTest(PyCoTestCase):
    def test_Fn(self):
        x = np.linspace(-3, 3, 101)
        f = GT.Fn(x, 0)
        self.assertArrayAlmostEqual(GT.Fn(x, 0), (1-erf(x/np.sqrt(2)))/2)
    def test_s(self):
        L = lambda x: np.where(x<=1, 2/np.pi*x*ellipk(x**2), 2/np.pi*ellipk(1/x**2))
        snum = lambda ξ: np.array([quad(L, 0, _ξ)[0] for _ξ in ξ])
        x = np.linspace(0.01, 5, 101)
        self.assertArrayAlmostEqual(GT.s(x), snum(x))
    def test_no_oscillating_solution(self):
        μ = 24.0425586841
        w1, p1, rho1 = GT.GreenwoodTripp(0.5, μ)
        w2, p2, rho2 = GT.GreenwoodTripp(1.0, μ)

        self.assertTrue(np.all(np.diff(p1)<0.0))
        self.assertTrue(np.all(np.diff(p2)<0.0))

        #import matplotlib.pyplot as plt
        #plt.subplot(1,2,1)
        #plt.plot(rho2, w2, 'k-')
        #plt.plot(rho1, w1, 'r-')
        #plt.subplot(1,2,2)
        #plt.plot(rho2, p2, 'k-')
        #plt.plot(rho1, p1, 'r-')
        #plt.yscale('log')
        #plt.show()
    def test_md_dmt_limit(self):
        A = np.linspace(0.001, 10, 11)
        N, d = MD._load_and_displacement(A, 1e-12)
        self.assertArrayAlmostEqual(N, A**3-2)
        self.assertArrayAlmostEqual(d, A**2, tol=1e-5)
    def test_md_jkr_limit(self):
        A = np.linspace(0.001, 10, 11)
        N, d = MD._load_and_displacement(A, 1e3)
        self.assertArrayAlmostEqual(N, A**3-A*np.sqrt(6*A), tol=1e-4)
        self.assertArrayAlmostEqual(d, A**2-2/3*np.sqrt(6*A), tol=1e-4)
