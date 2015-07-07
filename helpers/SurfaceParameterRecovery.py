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

import PyPyContact.Tools as Tools
import PyPyContact.Surface as Surf

from SurfAnalysisHelper import test_surf_analysis


def main():
    siz = 3
    size = (siz, siz)
    hurst = .9
    h_rms = 1
    res = 500
    resolution = (res, res)
    lam_max = .5
    for i in range(1):
        surf_gen = Tools.RandomSurfaceExact(resolution, size, hurst, h_rms, seed = i, lambda_max=lam_max)
        surf = surf_gen.get_surface(roll_off=0, lambda_max=lam_max)
        print("q_min = {}".format(2*np.pi/lam_max))
        h_rms_fromC_in = surf.compute_h_rms_fromReciprocSpace()
        #print("h_rms = {}".format(surf.h_rms()))

    print("input  values: H = {:.3f}, h_rms = {:.3f}, h_rms(C) = {:.3f}".format(hurst, h_rms, h_rms_fromC_in))

    surf_char = Tools.CharacterisePeriodicSurface(surf)
    hurst_out, prefactor_out = surf_char.estimate_hurst(full_output=True, lambda_max=lam_max)
    h_rms_out = surf_char.compute_h_rms()
    h_rms_fromC = surf_char.compute_h_rms_fromReciprocSpace()
    print("output values: H = {:.3f}, h_rms = {:.3f}, h_rms(C) = {:.3f}".format(hurst_out, h_rms_out, h_rms_fromC))
    q_min = 2*np.pi/size[0]
    beta = surf_gen.compute_prefactor()/np.sqrt(np.prod(size))
    print("alpha_in = {}, alpha_out = {}".format(beta, prefactor_out))
    ax = plt.figure().add_subplot(111)

    ax.loglog(surf_char.q, beta**2*surf_char.q**(-2*(hurst+1)), label="theoretical")

    ax.loglog(surf_char.q, 4*np.pi*hurst*h_rms**2 *surf_char.q**(-2-2*hurst)/(q_min**(-2*hurst)), label="lars")

    ax.loglog(surf_char.q, surf_char.C, alpha=.5, label='full')
    ax.loglog(surf_char.q, surf_char.q**(-2*(hurst_out+1))*prefactor_out, label="recovered", ls = '--')
    ax.legend(loc='best')
    ax.grid(True)
    ax.set_ylim(bottom=surf_char.C[-1]/10)
    print("prefactor_in, prefactor_out = {}, {}, rel_err = {}".format(beta, prefactor_out, abs(1-prefactor_out/beta)))




if __name__ == '__main__':
    main()
    plt.show()
