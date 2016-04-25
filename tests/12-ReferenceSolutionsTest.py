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

    from .pycotest import PyCoTestCase
    import PyCo.ReferenceSolutions.GreenwoodTripp as GT
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