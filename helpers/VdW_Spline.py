#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   VdW_Spline.py

@author Till Junge <till.junge@kit.edu>

@date   21 May 2015

@brief  Trying to figure out under what conditions we can use the 4th order
        spline to smoothen out a vdW potential

@section LICENCE

 Copyright (C) 2015 Till Junge

PyCo is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation, either version 3, or (at
your option) any later version.

PyCo is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with GNU Emacs; see the file COPYING. If not, write to the
Free Software Foundation, Inc., 59 Temple Place - Suite 330,
Boston, MA 02111-1307, USA.
"""

from PyCo.ContactMechanics import LJ93smooth
from sympy import Symbol, pprint
import sympy
from copy import deepcopy

r = Symbol('r', positive=True)
dr = Symbol('Δr', real=True)
dgam = Symbol('(Δγ-γ)', negative=True)
gam = Symbol('γ', positive=True)

c_sr = Symbol('C_sr', positive=True)
hama = Symbol('A', positive=True)

dV_t = Symbol('dV_t', real=True)
ddV_t = Symbol('ddV_t', real=True)


vdw = -hama/(12*r**2*sympy.pi)+c_sr/r**8
dvdw = sympy.diff(vdw, r)
ddvdw = sympy.diff(dvdw, r)


radicant = -12 * dgam * ddV_t + 9 * dV_t**2
print('radicant')
pprint(radicant)

print("radicant for van der Waals")
vdw_radicant = radicant.subs({dgam: vdw, dV_t:dvdw, ddV_t:ddvdw})
pprint(vdw_radicant)

#boundaries of validity
print("Roots of radicant")
sol_radicant = sympy.solve(vdw_radicant, r)
for sol in sol_radicant:
    pprint(sol)
    print('is positive: {}'.format(sol.is_positive))
#pprint(sol_radicant)
