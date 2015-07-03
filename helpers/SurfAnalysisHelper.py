#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   SurfAnalysisHelper.py

@author Till Junge <till.junge@kit.edu>

@date   25 Jun 2015

@brief  Comparisons between measured and generated surfaces and implications

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

import numpy as np
import matplotlib.pyplot as plt

import PyPyContact.Tools as Tools
import PyPyContact.Surface as Surf


def test_surf_analysis(surf, name):
    surf_char = Tools.CharacterisePeriodicSurface(surf)
    q = surf_char.q
    C = surf_char.C
    fig = plt.figure()
    ax=fig.add_subplot(111)
    ax.set_yscale('log')
    ax.set_xscale('log')
    #ax.set_ylim(bottom=1e-16)
    plt.loglog(q, C, alpha=.1)
    mean, err, q_g = surf_char.grouped_stats(100)
    ax.errorbar(q_g, mean, yerr=err)
    ax.set_title("{}: H={:.2e}, h_rms={:.2e}".format(name, surf_char.estimate_hurst(), np.sqrt((surf.profile()**2).mean())))
    a, b = np.polyfit(np.log(q), np.log(C), 1)
    ax.plot(q, q**a*np.exp(b))
    print(surf.profile().mean())
    print("(min, max)(C) : {}".format((C.min(), C.max())))

def main():
    size = (2000.e-9, 2000.e-9)
    hurst = .87
    h_rms = 13.8e-9
    lambda_max= 4e-9
    surf_dict = dict()
    surf_dict["Topo1_Fit"] = Surf.NumpyTxtSurface("SurfaceExample.asc", size=size, factor=1e-9)
    surf_dict["Topo1_NoFit"] = Surf.NumpyTxtSurface("SurfaceExampleUnfiltered.asc", size=size, factor=1e-9)
    resolution = surf_dict["Topo1_NoFit"].resolution
    surf_dict["exact"] = Tools.RandomSurfaceExact(resolution, size, hurst, h_rms).get_surface()
    surf_dict["Gauss"] = Tools.RandomSurfaceGaussian(resolution, size, hurst, h_rms).get_surface()

    for name, surf in surf_dict.items():
        test_surf_analysis(surf, name)
    print("done")

if __name__ == '__main__':
    main()
    plt.show()
