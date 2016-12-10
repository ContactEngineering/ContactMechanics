#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   __init__.py

@author Lars Pastewka <lars.pastewka@kit.edu>

@date   29 June 2016

@brief  Maugis-Dugdale cohesive zone model for a sphere contacting an elastic
        flat.
        See: D. Maugis, J. Colloid Interf. Sci. 150, 243 (1992)

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

def maugis_parameter(radius, elastic_modulus, work_of_adhesion,
                     cohesive_stress):
    K = 4*elastic_modulus/3
    return 2*cohesive_stress/(np.pi*work_of_adhesion*K**2/radius)**(1./3)

def fm(m, a, lam):
    mu = np.sqrt(m*m-1)
    tanmu = np.arctan(mu)
    # This is Eq. (6.17) from Maugis' paper
    return (1./2)*lam*a*a*(mu+(m*m-2)*tanmu)+(4./3)*lam*lam*a*(mu*tanmu-m+1)-1

def _load(a, lam, return_m=False):
    """
    Compute load for the Maugis-Dugdale model given the area and Maugis
    parameter.

    Parameters
    ----------
    a : array_like
        Non-dimensional contact radius
    lam : float
        Maugis parameter

    Returns
    -------
    P : array
        Non-dimensional load
    """
    m = afindroot(fm, 1.0, 1e12, a, lam)
    mu = np.sqrt(m*m-1)
    tanmu = np.arctan(mu)
    P = a*a*a-lam*a*a*(mu+m*m*tanmu)
    if return_m:
        return P, m
    else:
        return P

def _load_and_displacement(a, lam, return_m=False):
    """
    Compute load and displacement for the Maugis-Dugdale model given the area
    and Maugis parameter.

    Parameters
    ----------
    a : array_like
        Non-dimensional contact radius
    lam : float
        Maugis parameter

    Returns
    -------
    N : array
        Non-dimensional load
    d : array
        Non-dimensional displacement
    """
    P, m = _load(a, lam, return_m=True)
    mu = np.sqrt(m*m-1)
    d = a*a-(4./3)*a*lam*mu
    if return_m:
        return P, d, m
    else:
        return P, d

def load_and_displacement(contact_radius, radius, elastic_modulus,
                          work_of_adhesion, cohesive_stress):
    lam = maugis_parameter(radius, elastic_modulus, work_of_adhesion,
                          cohesive_stress)
    K = 4*elastic_modulus/3
    A = contact_radius/(np.pi*work_of_adhesion*radius**2/K)**(1./3)
    N, d = _load_and_displacement(A, lam)
    return (N*np.pi*work_of_adhesion*radius,
            d*(np.pi**2*work_of_adhesion**2*radius/K**2)**(1./3))
