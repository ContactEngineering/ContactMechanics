#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   __init__.py

@author Lars Pastewka <lars.pastewka@kit.edu>

@date   07 Mar 2017

@brief  Westergaard solution for partial contact of a sinoisoidal periodic
        indenter.
        See: H.M. Westergaard, Trans. ASME, J. Appl. Mech. 6, 49 (1939)

@section LICENCE

 Copyright (C) 2017 Lars Pastewka

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

###

def _pressure(x, contact_radius=None, mean_pressure=None):
    """
    Compute pressure for Westergaard solution of unit amplitude.

    Parameters
    ----------
    x : array_like
        Fractional positions array
    contact_radius : float
        Dimensionless contact radius
    mean_pressure : float
        Dimensionless mean pressure

    Returns
    -------
    p : array
        Non-dimensional pressure
    """
    psi = np.pi*x
    if contact_radius is not None:
        psia = np.pi*contact_radius
        mean_pressure = np.pi*np.sin(psia)**2
    elif mean_pressure is not None:
        psia = np.arcsin(np.sqrt(mean_pressure/np.pi))
    p = np.zeros_like(x)
    m = np.cos(psi) > np.cos(psia)
    if m.sum() > 0:
        p[m] = 2*np.cos(psi[m])/(np.sin(psia)**2)*np.sqrt(np.sin(psia)**2-np.sin(psi[m])**2)
    return p
