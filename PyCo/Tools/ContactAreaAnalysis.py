# -*- coding:utf-8 -*-
"""
@file   ContactAreaAnalysis.py

@author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>

@date   10 Apr 2017

@brief  Tool for analysis contact geometry

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
