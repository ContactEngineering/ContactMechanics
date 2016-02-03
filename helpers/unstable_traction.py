#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   unstable_traction.py

@author Till Junge <till.junge@kit.edu>

@date   31 Mar 2015

@brief  Trying to show that soft materials have unstable pulloff behavior

@section LICENCE

 Copyright (C) 2015 Till Junge

unstable_traction.py is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation, either version 3, or (at
your option) any later version.

unstable_traction.py is distributed in the hope that it will be useful, but
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

from PyCo.System import SystemFactory
from PyCo.Surface import Sphere
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
    surface = Sphere(radius, res, size, standoff=float('inf'))

    sigma = radius/10
    for factor in (.01, .1):#, 1.):
        epsilon = sigma * young*factor
        pot = LJ_pot(epsilon, sigma)
        system = SystemFactory(substrate, pot, surface)
        percent_min = .8
        offset = pot.r_min
        step = pot.r_min*.01
        pullof_forces = list()
        offsets = list()
        contact_area = list()
        disp = np.zeros(substrate.computational_resolution)
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
        ax1.set_title("factor = {}, resolution = {}".format(factor, res))
        plt.draw()

plt.ioff()
plt.show()
