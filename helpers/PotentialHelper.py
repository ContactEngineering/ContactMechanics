#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   PotentialHelper.py

@author Till Junge <till.junge@kit.edu>

@date   19 May 2015

@brief  symbolic calculations for potentials

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

from sympy import Symbol, pprint
import sympy


r = Symbol('r', positive=True)

def show_props(pot, name):
    print("{} potential:".format(name))
    #dpot = sympy.simplify(sympy.diff(pot, r))
    #ddpot = sympy.simplify(sympy.diff(dpot, r))
    dpot = sympy.diff(pot, r)
    ddpot = sympy.diff(dpot, r)
    pprint(pot)
    print("1st derivative:")
    pprint(dpot)
    print("2nd derivative:")
    pprint(ddpot)

    r_min = sympy.solve(dpot, r)
    print('r_min =')
    pprint(r_min[-1])

    r_infl = sympy.solve(ddpot, r)
    print('r_infl =')
    pprint(r_infl[-1])
    print()


def main():
    eps = Symbol('ε', positive=True)
    sig = Symbol('σ', positive=True)

    LJ93 = eps*(-(sig/r)**3 + (2*sig**9)/(15*r**9))
    show_props(LJ93, 'LJ')

    hama = Symbol("A", positive=True)
    c_sr = Symbol("C_sr", positive=True)
    vdW82 = -hama/(12*sympy.pi*r**2) + c_sr*r**-8
    show_props(vdW82, 'LJ')

if __name__ == '__main__':
    main()
