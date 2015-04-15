#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   hysteresis.py

@author Till Junge <till.junge@kit.edu>

@date   01 Apr 2015

@brief  simpre sphere-on-flat contact simulation to measure pull-off hysteresis

@section LICENCE

 Copyright (C) 2015 Till Junge

hysteresis.py is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation, either version 3, or (at
your option) any later version.

hysteresis.py is distributed in the hope that it will be useful, but
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

from PyPyContact.System import SystemFactory
from PyPyContact.Surface import Sphere
from PyPyContact.ContactMechanics import VDW82smoothMin as VdwPot
from PyPyContact.SolidMechanics import FreeFFTElasticHalfSpace as Substrate


plt.ion()

E_silicon = 166e9
nu_silicon = .17
E_diamond = 1220e9
nu_diamond = .2
young = 1./((1-nu_silicon**2)/E_silicon+(1-nu_diamond**2)/E_diamond)

radius = 40e-9
base_size = 2*radius
size = (base_size, base_size)

c_sr = 2.1e-78
hamaker = 68.1e-21
pot = VdwPot(c_sr, hamaker)

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

for base_res in (32, 64, 128, 256, 512):
    res = (base_res, base_res)

    substrate = Substrate(res, young, size)

    surface = Sphere(radius, res, size, standoff=float('inf'))
    system = SystemFactory(substrate, pot, surface)
    offset = pot.r_c
    step = pot.r_min*.01
    pullof_forces = list()
    offsets = list()
    contact_area = list()
    disp = np.zeros(substrate.computational_resolution)
    force = -1.

    def iterator(initial_offset):
        loc_offset = float(initial_offset)
        yield loc_offset
        while force <= 0.:
            loc_offset -= step
            yield loc_offset
        for i in range(3):
            loc_offset += step
            yield loc_offset
        while force < 0.:
            loc_offset += step
            yield loc_offset

    ax1.set_ylabel("normal force")
    ax2.set_ylabel("contact area", color='r')
    line_force, = ax1.plot(offsets, pullof_forces,
                           label="resolution = {}".format(res))
    line_area, = ax2.plot(offsets, contact_area, color=line_force.get_color(),
                          linestyle='--')
    ax1.legend(loc='center right')
    marker, = ax1.plot((), (), marker='+', color='b', markersize=20)

    for offset in iterator(offset):
        result = system.minimize_proxy(offset, disp, deproxify_everytime=False)
        disp = system.disp
        force = system.babushka.compute_normal_force()
        contact_area.append(system.babushka.compute_contact_area())
        pullof_forces.append(force)
        offsets.append(offset)
        line_force.set_xdata(offsets)
        line_force.set_ydata(pullof_forces)
        marker.set_ydata((force,))
        marker.set_xdata((offset,))
        line_area.set_xdata(offsets)
        line_area.set_ydata(contact_area)
        ax1.relim()
        ax1.autoscale_view()
        ax2.relim()
        ax2.autoscale_view()
        fig.canvas.draw()

    marker.set_ydata(())
    marker.set_xdata(())
    fig.savefig("fig_{:0>5}.png".format(res[0]))



plt.ioff()
plt.show()
