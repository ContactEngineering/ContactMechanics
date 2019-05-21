#
# Copyright 2018 Lars Pastewka
#           2015-2016 Till Junge
# 
# ### MIT license
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
"""
sets uf a random fractal surface and recovers the hurst exponent and h_rmso
"""
import numpy as np
import matplotlib.pyplot as plt

import PyCo.Tools as Tools
import PyCo.Topography as Surf

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
