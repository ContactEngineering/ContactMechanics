#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   __init__.py

@author Lars Pastewka <lars.pastewka@kit.edu>

@date   09 Dec 2016

@brief  Maugis-Dugdale (cohesive zone) model for a cylinder contacting an
        elastic flat.
        See: J.M. Baney, C.-Y. Hui, J. Adhesion Sci. Technol. 11, 393 (1997)

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

from __future__ import division

import numpy as np
from math import pi
from scipy.optimize import brentq

from PyCo.ReferenceSolutions.MaugisDugdale import afindroot

###

def maugis_parameter(radius, elastic_modulus, work_of_adhesion,
                     cohesive_stress):
    return 4*cohesive_stress/(np.pi**2*elastic_modulus**2*work_of_adhesion/radius)**(1/3)

### Dimensionless quantities

def fm(m, a, lam):
    mu = np.sqrt(m*m-1)
    # This is Eq. (23) from Baney and Hui's paper
    return lam*a*a/2*(m*mu-np.log(m+mu))+lam*lam*a/2*(mu*np.log(m+mu)-m*np.log(m))-1

def _load(a, lam, return_m=False):
    """
    Compute load for the Maugis-Dugdale model given the dimensionless contact
    radius and Maugis parameter.

    Parameters
    ----------
    a : array_like
        Dimensionless contact radius
    lam : float
        Maugis parameter

    Returns
    -------
    P : array
        Non-dimensional load
    """
    m = afindroot(fm, 1.0, 1e12, a, lam)
    mu = np.sqrt(m*m-1)
    P = a*a-lam*a*mu
    if return_m:
        return P, m
    else:
        return P

### Dimensional quantities

def load(contact_radius, radius, elastic_modulus, work_of_adhesion,
         cohesive_stress):
    lam = maugis_parameter(radius, elastic_modulus, work_of_adhesion,
                           cohesive_stress)
    a = contact_radius/(2*(work_of_adhesion*radius**2/(np.pi*elastic_modulus))**(1/3))
    P = _load(a, lam)
    return P*(np.pi*elastic_modulus*work_of_adhesion**2*radius)**(1/3)
