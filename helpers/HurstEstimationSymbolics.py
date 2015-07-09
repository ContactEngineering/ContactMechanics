#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   HurstEstimationSymbolics.py

@author Till Junge <till.junge@kit.edu>

@date   08 Jul 2015

@brief  objective function and jacobians for hurst estimation

@section LICENCE

 Copyright (C) 2015 Till Junge

HurstEstimationSymbolics.py is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation, either version 3, or (at
your option) any later version.

HurstEstimationSymbolics.py is distributed in the hope that it will be useful, but
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


q = Symbol('q', positive=True)
denom = Symbol('denom', positive=True)
C = Symbol('C', positive=True)
C0 = Symbol('C_0', positive=True)
H = Symbol('H', positive=True)
cor = Symbol('K', positive=True)


obj = (C-C0*q**(-2-2*H))**2/denom
pprint(obj)

jacob = [sympy.diff(obj, var) for var in (H, C0)]
print("jacobian:")
pprint(jacob)

sol_C0 = sympy.solve(jacob[1], C0)
pprint(sol_C0)

fprime_h = sympy.diff(jacob[0], H)

pprint(sympy.simplify(fprime_h))


pprint(sympy.diff(q**(-2-2*H), H))
