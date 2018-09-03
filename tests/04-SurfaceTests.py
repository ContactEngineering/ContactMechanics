#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   04-SurfaceTests.py

@author Till Junge <till.junge@kit.edu>

@date   27 Jan 2015

@brief  Tests surface classes

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
    import numpy.matlib as mp
    from numpy.random import rand, random
    import tempfile, os
    from tempfile import TemporaryDirectory as tmp_dir
    import os

    from PyCo.Topography import (NumpyTxtSurface, NumpyAscSurface, NumpyTopography, DetrendedTopography, Sphere,
                                 rms_height, rms_slope, compute_derivative, shift_and_tilt, read,
                                 read_asc, read_di, read_h5, read_hgt, read_ibw, read_mat, read_opd, read_x3p)
    from PyCo.Topography.FromFile import detect_format, get_unit_conversion_factor
    from PyCo.Goodies.SurfaceGeneration import RandomSurfaceGaussian

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

        S1 = NumpyTopography(h)
        with tmp_dir() as dir:
            fname = os.path.join(dir,"surface")
            S1.save(dir+"/surface")

            S2 = NumpyTxtSurface(fname)
        S3 = Sphere(R, (res, res), (l, l), (x_c, y_c))
        self.assertTrue(np.array_equal(S1.array(), S2.array()))
        self.assertTrue(np.array_equal(S1.array(), S3.array()), )

    def test_laplacian_estimation(self):
        a = np.random.rand()-.5
        b = np.random.rand()-.5
        laplacian = 2*(a+b)

        res = (5, 5)
        size = (8.5, 8.5)
        x = y = np.linspace(0, 8.5, 6)[:-1]
        X, Y = np.meshgrid(x, y)
        F = a*X**2 + b*Y**2
        surf = NumpyTopography(F, size=size)
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
    def test_example1(self):
        surf = NumpyAscSurface('tests/file_format_examples/example1.txt')
        self.assertEqual(surf.shape, (1024, 1024))
        self.assertAlmostEqual(surf.size[0], 2000)
        self.assertAlmostEqual(surf.size[1], 2000)
        self.assertAlmostEqual(surf.rms_height(), 17.22950485567042)
        self.assertAlmostEqual(rms_slope(surf), 0.45604053876290829)
        self.assertEqual(surf.unit, 'nm')
    def test_example2(self):
        surf = read_asc('tests/file_format_examples/example2.txt')
        self.assertEqual(surf.shape, (650, 650))
        self.assertAlmostEqual(surf.size[0], 0.0002404103)
        self.assertAlmostEqual(surf.size[1], 0.0002404103)
        self.assertAlmostEqual(surf.rms_height(), 2.7722350402740072e-07)
        self.assertAlmostEqual(rms_slope(surf), 0.35157901772258338)
        self.assertEqual(surf.unit, 'm')
    def test_example3(self):
        surf = read_asc('tests/file_format_examples/example3.txt')
        self.assertEqual(surf.shape, (256, 256))
        self.assertAlmostEqual(surf.size[0], 10e-6)
        self.assertAlmostEqual(surf.size[1], 10e-6)
        self.assertAlmostEqual(surf.rms_height(), 3.5222918750198742e-08)
        self.assertAlmostEqual(rms_slope(surf), 0.19231536279425226)
        self.assertEqual(surf.unit, 'm')
    def test_example4(self):
        surf = read_asc('tests/file_format_examples/example4.txt')
        self.assertEqual(surf.shape, (305, 75))
        self.assertAlmostEqual(surf.size[0], 0.00011280791)
        self.assertAlmostEqual(surf.size[1], 2.773965e-05)
        self.assertAlmostEqual(surf.rms_height(), 1.1745891510991089e-07)
        self.assertAlmostEqual(surf.rms_height(kind='Rq'), 1.1745891510991089e-07)
        self.assertAlmostEqual(rms_slope(surf), 0.067915823359553706)
        self.assertEqual(surf.unit, 'm')

class DetrendedSurfaceTest(unittest.TestCase):
    def setUp(self):
        pass
    def test_smooth_flat(self):
        a = 1.2
        b = 2.5
        d = .2
        arr = np.arange(5)*a+d
        arr = arr + np.arange(6).reshape((-1, 1))*b
        surf = DetrendedTopography(NumpyTopography(arr), detrend_mode='slope')
        self.assertAlmostEqual(surf[...].mean(), 0)
        self.assertAlmostEqual(rms_slope(surf), 0)
        surf = DetrendedTopography(NumpyTopography(arr), detrend_mode='height')
        self.assertAlmostEqual(surf[...].mean(), 0)
        self.assertAlmostEqual(rms_slope(surf), 0)
        self.assertTrue(rms_height(surf) < rms_height(arr))
        surf2 = DetrendedTopography(NumpyTopography(arr, size=(1, 1)), detrend_mode='height')
        self.assertAlmostEqual(rms_slope(surf2), 0)
        self.assertTrue(rms_height(surf2) < rms_height(arr))
        self.assertAlmostEqual(rms_height(surf), rms_height(surf2))
    def test_smooth_curved(self):
        a = 1.2
        b = 2.5
        c = 0.1
        d = 0.2
        e = 0.3
        f = 5.5
        x = np.arange(5).reshape((1, -1))
        y = np.arange(6).reshape((-1, 1))
        arr = f+x*a+y*b+x*x*c+y*y*d+x*y*e
        surf = DetrendedTopography(NumpyTopography(arr, size=(3., 2.5)), detrend_mode='curvature')
        self.assertAlmostEqual(surf.coeffs[0], -2*b)
        self.assertAlmostEqual(surf.coeffs[1], -2*a)
        self.assertAlmostEqual(surf.coeffs[2], -4*d)
        self.assertAlmostEqual(surf.coeffs[3], -4*c)
        self.assertAlmostEqual(surf.coeffs[4], -4*e)
        self.assertAlmostEqual(surf.coeffs[5], -f)
        self.assertAlmostEqual(surf.rms_height(), 0.0)
        self.assertAlmostEqual(surf.rms_slope(), 0.0)

    def test_randomly_rough(self):
        surface = RandomSurfaceGaussian((512, 512), (1., 1.), 0.8, rms_height=1).get_surface()
        cut = NumpyTopography(surface[:64, :64], size=(64., 64.))
        untilt1 = DetrendedTopography(cut, detrend_mode='height')
        untilt2 = DetrendedTopography(cut, detrend_mode='slope')
        self.assertTrue(untilt1.rms_height() < untilt2.rms_height())
        self.assertTrue(untilt1.rms_slope() > untilt2.rms_slope())

class detectFormatTest(unittest.TestCase):
    def setUp(self):
        pass
    def test_detection(self):
        self.assertEqual(detect_format('tests/file_format_examples/example1.di'), 'di')
        self.assertEqual(detect_format('tests/file_format_examples/example2.di'), 'di')
        self.assertEqual(detect_format('tests/file_format_examples/example.ibw'), 'ibw')
        self.assertEqual(detect_format('tests/file_format_examples/example.opd'), 'opd')
        self.assertEqual(detect_format('tests/file_format_examples/example.x3p'), 'x3p')
        self.assertEqual(detect_format('tests/file_format_examples/example1.mat'), 'mat')

class matSurfaceTest(unittest.TestCase):
    def setUp(self):
        pass
    def test_read(self):
        surface = read_mat('tests/file_format_examples/example1.mat')
        nx, ny = surface.shape
        self.assertEqual(nx, 2048)
        self.assertEqual(ny, 2048)
        self.assertAlmostEqual(surface.rms_height(), 1.234061e-07)

class x3pSurfaceTest(unittest.TestCase):
    def setUp(self):
        pass
    def test_read(self):
        surface = read_x3p('tests/file_format_examples/example.x3p')
        nx, ny = surface.shape
        self.assertEqual(nx, 777)
        self.assertEqual(ny, 1035)
        sx, sy = surface.size
        self.assertAlmostEqual(sx, 0.00068724)
        self.assertAlmostEqual(sy, 0.00051593)
        surface = read_x3p('tests/file_format_examples/example2.x3p')
        nx, ny = surface.shape
        self.assertEqual(nx, 650)
        self.assertEqual(ny, 650)
        sx, sy = surface.size
        self.assertAlmostEqual(sx, 8.29767313942749e-05)
        self.assertAlmostEqual(sy, 0.0002044783737930349)

class opdSurfaceTest(unittest.TestCase):
    def setUp(self):
        pass
    def test_read(self):
        surface = read_opd('tests/file_format_examples/example.opd')
        nx, ny = surface.shape
        self.assertEqual(nx, 640)
        self.assertEqual(ny, 480)
        sx, sy = surface.size
        self.assertAlmostEqual(sx, 0.125909140)
        self.assertAlmostEqual(sy, 0.094431855)

class diSurfaceTest(unittest.TestCase):
    def setUp(self):
        pass
    def test_read(self):
        # All units are nm
        for (fn, n, s, rmslist) in [
            ('example1.di', 512, 500.0, [9.9459868005603909, # Height
                                         114.01328027385664, # Height
                                         None, # Phase
                                         None]), # AmplitudeError
            ('example2.di', 512, 300.0, [24.721922008645919, # Height
                                         24.807150576054838, # Height
                                         0.13002312109876774]), # Deflection
            ('example3.di', 256, 10000.0, [226.42539668457405, # ZSensor
                                           None, # AmplitudeError
                                           None, # Phase
                                           264.00285276203158]), # Height
            ('example4.di', 512, 10000.0, [81.622909804184744, # ZSensor
                                           0.83011806260022758, # AmplitudeError
                                           None]) # Phase
            ]:
            surfaces = read_di('tests/file_format_examples/{}'.format(fn))
            if type(surfaces) is not list:
                surfaces = [surfaces]
            for surface, rms in zip(surfaces, rmslist):
                nx, ny = surface.shape
                self.assertEqual(nx, n)
                self.assertEqual(ny, n)
                sx, sy = surface.size
                if type(surface.unit) is tuple:
                    unit, dummy = surface.unit
                else:
                    unit = surface.unit
                self.assertAlmostEqual(sx*get_unit_conversion_factor(unit, 'nm'), s)
                self.assertAlmostEqual(sy*get_unit_conversion_factor(unit, 'nm'), s)
                if rms is not None:
                    self.assertAlmostEqual(surface.rms_height(), rms)
                    self.assertEqual(unit, 'nm')

class ibwSurfaceTest(unittest.TestCase):
    def setUp(self):
        pass
    def test_read(self):
        surface = read_ibw('tests/file_format_examples/example.ibw')
        nx, ny = surface.shape
        self.assertEqual(nx, 512)
        self.assertEqual(ny, 512)
        sx, sy = surface.size
        self.assertAlmostEqual(sx, 5.00978e-8)
        self.assertAlmostEqual(sy, 5.00978e-8)
        self.assertEqual(surface.unit, 'm')
    def test_detect_format_then_read(self):
        f = open('tests/file_format_examples/example.ibw', 'rb')
        fmt = detect_format(f)
        self.assertTrue(fmt, 'ibw')
        surface = read(f, format=fmt)
        f.close()

class hgtSurfaceTest(unittest.TestCase):
    def setUp(self):
        pass
    def test_read(self):
        surface = read_hgt('tests/file_format_examples/N46E013.hgt')
        nx, ny = surface.shape
        self.assertEqual(nx, 3601)
        self.assertEqual(ny, 3601)

class h5SurfaceTest(unittest.TestCase):
    def setUp(self):
        pass
    def test_detect_format_then_read(self):
        self.assertEqual(detect_format('tests/file_format_examples/surface.2048x2048.h5'), 'h5')
    def test_read(self):
        surface = read_h5('tests/file_format_examples/surface.2048x2048.h5')
        nx, ny = surface.shape
        self.assertEqual(nx, 2048)
        self.assertEqual(ny, 2048)
