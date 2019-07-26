#
# Copyright 2018-2019 Antoine Sanner
#           2018-2019 Lars Pastewka
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
Trying to show that soft materials have unstable pulloff behavior
"""

import numpy as np
import matplotlib.pyplot as plt

from PyCo.System import make_system
from PyCo.Topography import make_sphere
from PyCo.ContactMechanics import LJ93smoothMin as LJ_pot
from PyCo.SolidMechanics import FreeFFTElasticHalfSpace as Substrate

plt.ion()
for base_res in (64, ):#128, 256, 512, 1024):
    res = (base_res, base_res)
    base_size = 128.
    size = (base_size, base_size)
    young = 10

    substrate = Substrate(res, young, size)

    radius = size[0]/10
    surface = make_sphere(radius, res, size, standoff=float('inf'))

    sigma = radius/10
    for factor in (.01, .1):#, 1.):
        epsilon = sigma * young*factor
        pot = LJ_pot(epsilon, sigma)
        system = make_system(substrate, pot, surface)
        percent_min = .8
        offset = pot.r_min
        step = pot.r_min*.01
        pullof_forces = list()
        offsets = list()
        contact_area = list()
        disp = np.zeros(substrate.nb_domain_grid_pts)
        force = -1.
        while force < 0:
            result = system.minimize_proxy(offset, disp)
            disp = system.disp
            force = system.compute_normal_force()
            contact_area.append(system.compute_nb_contact_pts(disp, offset))
            print("force = {}".format(force))
            pullof_forces.append(force/epsilon)
            offset += step
            offsets.append(offset)

        fig, ax1 = plt.subplots()
        ax1.plot(offsets, pullof_forces)
        ax1.set_ylabel("normal force/eps")
        ax2 = ax1.twinx()
        ax2.plot(offsets, contact_area, color='r')
        ax2.set_ylabel("# contact_pts", color='r')
        ax1.set_title("factor = {}, nb_grid_pts = {}".format(factor, res))
        plt.draw()

plt.ioff()
plt.show()
