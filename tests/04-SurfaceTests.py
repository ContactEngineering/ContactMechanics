#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   04-SurfaceTests.py

@author Till Junge <till.junge@kit.edu>

@date   27 Jan 2015

@brief  Tests surface classes

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
    import numpy.matlib as mp
    from numpy.random import rand, random
    import tempfile, os
    from tempfile import TemporaryDirectory as tmp_dir
    import os

    from PyPyContact.Surface import NumpyTxtSurface, NumpySurface, Sphere
    from PyPyContact.Surface import NumpyAscSurface, read_asc
    from PyPyContact.Tools import compute_rms_slope

except ImportError as err:
    import sys
    print(err)
    sys.exit(-1)


class NumpyTxtSurfaceTest(unittest.TestCase):
    def setUp(self):
        pass
    def test_saving_loading_and_sphere(self):
        l = 8+4*rand()  # domain size (edge lenght of square)
        R = 17+6*rand() # sphere radius
        res = 2        # resolution
        x_c = l*rand()  # coordinates of center
        y_c = l*rand()
        x = np.arange(res, dtype = float)*l/res-x_c
        y = np.arange(res, dtype = float)*l/res-y_c
        r2 = np.zeros((res, res))
        for i in range(res):
            for j in range(res):
                r2[i,j] = x[i]**2 + y[j]**2
        h = np.sqrt(R**2-r2)-R # profile of sphere

        S1 = NumpySurface(h)
        with tmp_dir() as dir:
            fname = os.path.join(dir,"surface")
            S1.save(dir+"/surface")

            S2 = NumpyTxtSurface(fname)
        S3 = Sphere(R, (res, res), (l, l), (x_c, y_c))
        self.assertTrue(np.array_equal(S1.profile(), S2.profile()))
        self.assertTrue(np.array_equal(S1.profile(), S3.profile()),)

    def test_laplacian_estimation(self):
        a = np.random.rand()-.5
        b = np.random.rand()-.5
        laplacian = 2*(a+b)

        res = (5, 5)
        size = (8.5, 8.5)
        x = y = np.linspace(0, 8.5, 6)[:-1]
        X, Y = np.meshgrid(x, y)
        F = a*X**2 + b*Y**2
        surf = NumpySurface(F, size=size)
        L = np.zeros_like(F)
        for i in range(L.shape[0]):
            for j in range(L.shape[1]):
                L[i,j] = surf.estimate_laplacian((i, j))
        tol = 1e-10
        self.assertTrue(
            (abs(L-laplacian)).max() < tol,
            "Fail: the array should only contain the value {}, but it is \n{}.\nThe array was \n{}".format(laplacian, L, F))

class NumpyAscSurfaceTest(unittest.TestCase):
    def setUp(self):
        pass
    def test_format1(self):
        surf = NumpyAscSurface('tests/file_format_examples/format1.asc')
        self.assertEqual(surf.shape, (1024, 1024))
        self.assertAlmostEqual(surf.size[0], 2e-6)
        self.assertAlmostEqual(surf.size[1], 2e-6)
        self.assertAlmostEqual(surf.compute_rms_height(), 1.72295048557e-08)
        self.assertAlmostEqual(compute_rms_slope(surf), 0.45604053876290829)
    def test_format2(self):
        surf = read_asc('tests/file_format_examples/format2.txt')
        self.assertEqual(surf.shape, (650, 650))
        self.assertAlmostEqual(surf.size[0], 0.0002404103)
        self.assertAlmostEqual(surf.size[1], 0.0002404103)
        self.assertAlmostEqual(surf.compute_rms_height(), 2.7722350402740072e-07)
        self.assertAlmostEqual(compute_rms_slope(surf), 0.35157901772258338)

