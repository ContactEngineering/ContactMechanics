# -*- coding:utf-8 -*-
"""
@file   ScanningProbe.py

@author Lars Pastewka <lars.pastewka@kit.edu>

@date   06 Mar 2017

@brief  Emulation of scanning probe techniques

@section LICENCE

Copyright 2015-2017 Till Junge, Lars Pastewka

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np

from PyCo.Topography import NumpyTopography
from PyCo.Topography.common import _get_size

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

    profile = surface.array()
    nx, ny = surface.shape
    sx, sy = _get_size(surface)
    return NumpyTopography(_scan_topography(nx, ny, profile, tip_radius*nx/sx),
                           size=(sx, sy))
