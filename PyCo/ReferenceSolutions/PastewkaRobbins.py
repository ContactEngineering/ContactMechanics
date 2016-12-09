#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   __init__.py

@author Lars Pastewka <lars.pastewka@kit.edu>

@date   09 Dec 2016

@brief  Maugis-Dugdale (cohesive zone) model for a wedge contacting an elastic
        flat.
        See: L. Pastewka, M.O. Robbins, unpublished

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

def maugis_parameter(slope, elastic_modulus, cohesive_stress):
    return 2*cohesive_stress/(slope*elastic_modulus)

def fm(m, a, lam):
    mu = np.sqrt(m*m-1)
    return a*lam*(np.pi/2*m-m*np.arcsin(1/m)+np.log(m+mu)*(lam*mu-1)+lam*m*np.log(m))-1

### Dimensionless quantities

def _cohesive_zone(a, lam):
    """
    Returns the width of the cohesive zone m=b/a, where b is the cohesive zone
    edge and a is the contact radius.

    Parameters
    ----------
    a : array_like
        Dimensionless contact radius
    lam : float
        Maugis parameter

    Returns
    -------
    P : array
        Cohesive zone width
    """
    return afindroot(fm, 1.0, 1e12, a, lam)

def _load(a, lam, return_cohesive_zone=False):
    """
    Compute load for the Maugis-Dugdale model given the dimensionless contact
    radius and Maugis parameter.

    Parameters
    ----------
    a : array_like
        Dimensionless contact radius
    lam : float
        Maugis parameter
    return_cohesize_zone : bool
        Return width of cohesive zone if set to true

    Returns
    -------
    P : array
        Non-dimensional load
    m : array, optional
        Width of cohesive zone
    """
    m = _cohesive_zone(a, lam)
    mu = np.sqrt(m*m-1)
    P = a*(1-lam*mu)
    if return_cohesive_zone:
        return P, m
    else:
        return P

def _contact_radius(P, lam, return_cohesive_zone=False):
    a = afindroot(lambda a, P, lam: _load(a, lam)-P, 1e-6, 1e12, P, lam)
    if return_cohesive_zone:
        return a, _cohesive_zone(a, lam)
    else:
        return a

def _pressure(x, m, lam):
    p = np.zeros_like(x)
    p[:] = np.nan
    mask = abs(x)<1
    p[mask] = np.log((1+np.sqrt(1-x[mask]*x[mask]))/abs(x[mask]))-lam*np.arctan(np.sqrt((m*m-1)/(1-x[mask]*x[mask])))
    mask = np.logical_and(abs(x)>=1, abs(x)<=m)
    p[mask] = -np.pi*lam/2
    return p

### Dimensional quantities

def load(contact_radius, slope, elastic_modulus, work_of_adhesion,
         cohesive_stress):
    lam = maugis_parameter(slope, elastic_modulus, cohesive_stress)
    a = slope**2*contact_radius*elastic_modulus/(np.pi*work_of_adhesion)
    P = _load(a, lam)
    return P*work_of_adhesion*elastic_modulus/(2*cohesive_stress)
