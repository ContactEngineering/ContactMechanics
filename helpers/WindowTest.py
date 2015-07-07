#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   WindowTest.py

@author Till Junge <till.junge@kit.edu>

@date   07 Jul 2015

@brief  Compare windowed parameter recoups to non-windowed

@section LICENCE

 Copyright (C) 2015 Till Junge

WindowTest.py is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation, either version 3, or (at
your option) any later version.

WindowTest.py is distributed in the hope that it will be useful, but
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

def process(surf_name, surface):
    surfs = dict()
    surfs["Hanning "] = Tools.CharacteriseSurface(surface)
    surfs["Kaiser 8.6"] = Tools.CharacteriseSurface(surface, 'kaiser', {"beta": 8.6})
    surfs["periodic"] = Tools.CharacterisePeriodicSurface(surface)
    fig = plt.figure()
    ax=fig.add_subplot(111)
    ax.set_yscale('log')
    ax.set_xscale('log')
    colors = ('r', 'b', 'g')
    q_max = 2e8
    lambda_min = 2*np.pi/q_max
    for color, (name, surf) in zip(colors, surfs.items()):
        q = surf.q
        C = surf.C
        H, alpha = surf.estimate_hurst(lambda_min=lambda_min, full_output=True)
        plt.loglog(q, C, alpha=.1, color=color)
        mean, err, q_g = surf.grouped_stats(100)

        ax.errorbar(q_g, mean, yerr=err, color=color)
        ax.set_title("{}: H={:.2f}, h_rms={:.2e}".format(surf_name, H, np.sqrt((surface.profile()**2).mean())))
        a, b = np.polyfit(np.log(q), np.log(C), 1)
        ax.plot(q, q**(-2-2*H)*alpha, label="{}, H={:.2f}".format(name, H), color=color)
    ax.legend(loc='best')
    ax.grid(True)

def main():
    siz = 2000e-9
    size = (siz, siz)
    surface = Surf.NumpyTxtSurface("SurfaceExampleUnfiltered.asc", size=size, factor=1e-9)
    surfs = []
    surfs.append(('Topo1', surface))

    hurst = .98
    res = surface.resolution
    h_rms = 3.24e-8

    surface = Tools.RandomSurfaceGaussian(res, size, hurst, h_rms).get_surface()
    surfs.append(('Gauss periodic', surface))

    dsize = (2*siz, 2*siz)
    dres = (res[0], res[0])
    dsurface = Tools.RandomSurfaceGaussian(dres, dsize, hurst, h_rms).get_surface()
    surface = Surf.NumpySurface(dsurface.profile()[:res[0], :res[0]], size = size)
    surfs.append(('Gauss aperiodic', surface))

    for name, surf in surfs:
        process(name, surf)

if __name__ == "__main__":
    main()
    plt.show()
