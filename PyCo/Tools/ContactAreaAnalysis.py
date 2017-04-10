# -*- coding:utf-8 -*-
"""
@file   ContactAreaAnalysis.py

@author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>

@date   10 Apr 2017

@brief  Tool for analysis contact geometry

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

import numpy as np

from _PyCo import assign_patch_numbers, assign_segment_numbers, distance_map

###

# Stencils for determining nearest-neighbor relationships on a square grid
nn_stencil = [(1,0), (0,1), (-1,0), (0,-1)]
nnn_stencil = [(1,1), (1,0), (1,-1), (0,-1), (-1,-1), (-1,0), (-1,1), (0,1)]

###

def outer_perimeter(c, stencil=nn_stencil):
    """
    Return a map where the outer perimeter is marked with the patch number
    """

    c_nearby = np.zeros_like(c, dtype=bool)
    for dx, dy in stencil:
        tmp = c.copy()
        if dx != 0:
            tmp = np.roll(tmp, dx, 0)
        if dy != 0:
            tmp = np.roll(tmp, dy, 1)
        c_nearby = np.logical_or(c_nearby, tmp)
    return np.logical_and(np.logical_not(c), c_nearby)


def inner_perimeter(patch_ids, stencil=nn_stencil):
    """
    Return a map where the inner perimeter is marked with the patch number
    """

    c = outer_perimeter(patch_ids == 0, stencil)
    return c*patch_ids


def patch_areas(patch_ids):
    """
    Return a list containing patch areas
    """

    return np.bincount(patch_ids.reshape((-1,)))[1:]
