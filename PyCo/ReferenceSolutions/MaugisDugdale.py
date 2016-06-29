#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   __init__.py

@author Lars Pastewka <lars.pastewka@kit.edu>

@date   29 June 2016

@brief  Helper tools for PyCo

@section LICENCE

 Copyright (C) 2016 Lars Pastewka

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

import math

import numpy as np

###

import numpy as np
from math import pi
from scipy.optimize import brentq

def afindroot(f, left, right, a, b):
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        res = [ ]
        for ia, ib in zip(a, b):
            res += [ brentq(f, left, right, args=( ia, ib )) ]
        return np.asarray(res)
    elif isinstance(a, np.ndarray):
        res = [ ]
        for ia in a:
            res += [ brentq(f, left, right, args=( ia, b )) ]
        return np.asarray(res)
    else:
        return brentq(f, left, right, args=( a, b ))

###

def fA(m, A, lam):
    mu = np.sqrt(m*m-1)
    tanmu = np.arctan(mu)
    return (1./2)*lam*A*A*(mu+(m*m-2)*tanmu)+(4./3)*lam*lam*A*(mu*tanmu-m+1)-1

def load_and_displacement(A, lam, return_m=False):
    print fA(1.0, A, lam), fA(1e12, A, lam)
    m = afindroot(fA, 1.0, 1e12, A, lam)
    mu = np.sqrt(m*m-1)
    tanmu = np.arctan(mu)
    N = A*A*A-lam*A*A*(mu+m*m*tanmu)
    d = A*A-(4./3)*A*lam*mu
    if return_m:
        return N, d, m
    else:
        return N, d

