#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   simple_smooth.py

@author Till Junge <till.junge@kit.edu>

@date   21 May 2015

@brief  simplified smoothing for potentials

@section LICENCE

 Copyright (C) 2015 Till Junge

simple_smooth.py is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation, either version 3, or (at
your option) any later version.

simple_smooth.py is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with GNU Emacs; see the file COPYING. If not, write to the
Free Software Foundation, Inc., 59 Temple Place - Suite 330,
Boston, MA 02111-1307, USA.
"""

import numpy as np
from PyPyContact.ContactMechanics import VDW82


import matplotlib.pyplot as plt

hamaker = 68.1e-21
c_sr = 2.1e-78*1e-6
print(c_sr)

pot = VDW82(c_sr, hamaker)

r_c = 5e-10
r = np.linspace(.7*pot.r_min, 2*r_c, 1000)

V, forces, ddV = pot.evaluate(r, pot=True, forces=True, curb=True)
dV = -forces

deltaV, forces, deltaddV = [-float(bla) for bla in pot.evaluate(r_c, pot=True, forces=True, curb=True)]
deltadV = -forces

deltadV -= deltaddV*r_c

deltaV -= deltaddV/2*r_c**2 + deltadV*r_c

print("a, b, c = ({0})".format((deltaV, deltadV, deltaddV)))



fig = plt.figure()

ax = fig.add_subplot(311)
ax.grid(True)
plt.plot(r, V, label='original')
V1 = V + deltaddV/2*r**2
plt.plot(r, V1, label='step 1')
V2 = V1 + deltadV*r
plt.plot(r, V2, label='step 2')
V3 = V2 + deltaV
plt.plot(r, V3, label='step 3')
ax.set_ylim(bottom=1.1*V2.min(), top = -1.1*V2.min())
ax.legend(loc='best')

ax = fig.add_subplot(312)
ax.grid(True)
plt.plot(r, dV)
dV1 = dV + deltaddV*r
plt.plot(r, dV1)
dV2 = dV1 + deltadV
plt.plot(r, dV2)
ax.set_ylim(bottom=-1.1*dV1.max(), top = 1.1*dV1.max())

ax = fig.add_subplot(313)
ax.grid(True)
plt.plot(r, ddV)
plt.plot(r, ddV+deltaddV)
ax.set_ylim(bottom=1.1*ddV.min(), top = -1.1*ddV.min())


plt.show()
