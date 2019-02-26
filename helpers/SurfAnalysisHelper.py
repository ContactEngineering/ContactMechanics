#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   SurfAnalysisHelper.py

@author Till Junge <till.junge@kit.edu>

@date   25 Jun 2015

@brief  Comparisons between measured and generated surfaces and implications

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


def test_surf_analysis(surf, name):
    surf_char = Tools.CharacteriseSurface(surf)
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
    ax.set_title("{}: H={:.2e}, h_rms={:.2e}".format(name, surf_char.estimate_hurst(), np.sqrt((surf.heights() ** 2).mean())))
    a, b = np.polyfit(np.log(q), np.log(C), 1)
    ax.plot(q, q**a*np.exp(b))
    print(surf.heights().mean())
    print("(min, max)(C) : {}".format((C.min(), C.max())))

def main():
    size = (2000.e-9, 2000.e-9)
    hurst = .87
    h_rms = 13.8e-9
    lambda_max= 4e-9
    surf_dict = dict()
    surf_dict["Topo1_Fit"] = Surf.read_matrix("SurfaceExample.asc", size=size, factor=1e-9)
    surf_dict["Topo1_NoFit"] = Surf.read_matrix("SurfaceExampleUnfiltered.asc", size=size, factor=1e-9)
    resolution = surf_dict["Topo1_NoFit"].resolution
    surf_dict["exact"] = Tools.RandomSurfaceExact(resolution, size, hurst, h_rms).get_surface()
    surf_dict["Gauss"] = Tools.RandomSurfaceGaussian(resolution, size, hurst, h_rms).get_surface()

    for name, surf in surf_dict.items():
        test_surf_analysis(surf, name)
    print("done")

if __name__ == '__main__':
    main()
    plt.show()
