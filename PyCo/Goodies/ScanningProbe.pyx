# -*- coding:utf-8 -*-
"""
@file   ScanningProbe.py

@author Lars Pastewka <lars.pastewka@kit.edu>

@date   06 Mar 2017

@brief  Emulation of scanning probe techniques

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

import numpy as np

from PyCo.Topography import NumpyTopography
from PyCo.Tools import _get_size

###

cdef _scan_topography(int nx, int ny, double[:, :] surface, double tip_radius):
    scanned_surface = np.zeros([nx, ny])
    for i in range(nx):
        for j in range(ny):
            dx = ((((np.arange(nx)-i)+nx//2)%nx)-nx//2)
            dy = ((((np.arange(ny)-j)+ny//2)%ny)-ny//2)
            dx.shape = -1, 1
            dy.shape = 1, -1
            scanned_surface[i, j] = \
                (surface - (dx**2 + dy**2)/(2*tip_radius)).max()
    return scanned_surface

###

cpdef scan_surface(surface, tip_radius):
    """
    Scan surface with a power-law shaped tip.
    """

    profile = surface.profile()
    nx, ny = surface.shape
    sx, sy = _get_size(surface)
    return NumpyTopography(_scan_topography(nx, ny, profile, tip_radius*nx/sx),
                           size=(sx, sy))
