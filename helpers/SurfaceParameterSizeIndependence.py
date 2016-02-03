#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   SurfaceParameterRecovery.py

@author Till Junge <till.junge@kit.edu>

@date   30 Jun 2015

@brief  sets uf a random fractal surface and recovers the hurst exponent and h_rmso

@section LICENCE

 Copyright (C) 2015 Till Junge

SurfaceParameterRecovery.py is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation, either version 3, or (at
your option) any later version.

SurfaceParameterRecovery.py is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with GNU Emacs; see the file COPYING. If not, write to the
Free Software Foundation, Inc., 59 Temple Place - Suite 330,
Boston, MA 02111-1307, USA.
"""
import numpy as np
import matplotlib.pyplot as plt

import PyCo.Tools as Tools
import PyCo.Surface as Surf

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
        space_integral = (surf.profile()**2).sum()*area/nb_pts
        H = area/nb_pts*np.fft.fftn(surf.profile())
        frequ_integral = (np.conj(H)*H).sum().real/np.prod(siz)
        print("Parseval: {} =?= {}, rel_error = {}".format(space_integral, frequ_integral, abs(1-space_integral/frequ_integral)))


if __name__ == "__main__":
    main()
