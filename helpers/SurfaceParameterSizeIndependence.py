#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   SurfaceParameterRecovery.py

@author Till Junge <till.junge@kit.edu>

@date   30 Jun 2015

@brief  sets uf a random fractal surface and recovers the hurst exponent and h_rmso

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
import numpy as np
import matplotlib.pyplot as plt

import PyCo.Tools as Tools
import PyCo.Topography as Surf

from SurfAnalysisHelper import test_surf_analysis


def main():
    cases = ((3,  50),
             (3, 100),
             (6, 100))
    for siz, res in cases:
        res = (res, res)
        siz = (siz, siz)

        hurst = .7
        h_rms = 1.5
        lam_max = .5

        surf_gen = Tools.RandomSurfaceExact(res, siz, hurst, h_rms, lambda_max=lam_max)
        surf = surf_gen.get_surface(roll_off=0, lambda_max=lam_max)
        print("h_rms = {}".format(surf.h_rms()))
        h_rms_fromC_in = 2*np.pi*np.sqrt((abs(surf_gen.active_coeffs)**2/4/np.pi**2).sum())
        area = np.prod(siz)
        nb_pts = np.prod(res)
        space_integral = (surf.array() ** 2).sum() * area / nb_pts
        H = area/nb_pts*np.fft.fftn(surf.array())
        frequ_integral = (np.conj(H)*H).sum().real/np.prod(siz)
        print("Parseval: {} =?= {}, rel_error = {}".format(space_integral, frequ_integral, abs(1-space_integral/frequ_integral)))


if __name__ == "__main__":
    main()
