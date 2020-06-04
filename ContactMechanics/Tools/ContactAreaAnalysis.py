#
# Copyright 2017-2018, 2020 Lars Pastewka
#           2019 Antoine Sanner
#
# ### MIT license
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

"""
Tool for analysis contact geometry
"""

import numpy as np

from _SurfaceTopography import assign_patch_numbers, assign_segment_numbers, \
    distance_map, closest_patch_map  # noqa: F401

# Stencils for determining nearest-neighbor relationships on a square grid
nn_stencil = [(1, 0), (0, 1), (-1, 0), (0, -1)]
nnn_stencil = [(1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1),
               (0, 1)]


def coordination(c, stencil=nn_stencil):
    """
    Return a map with coordination numbers, i.e. number of neighboring patches
    that also contact
    """

    coordination = np.zeros_like(c, dtype=int)
    for dx, dy in stencil:
        tmp = np.array(c, dtype=bool, copy=True)
        if dx != 0:
            tmp = np.roll(tmp, dx, 0)
        if dy != 0:
            tmp = np.roll(tmp, dy, 1)
        coordination += tmp
    return coordination


def edge_perimeter_length(c, stencil=nn_stencil):
    """
    Return the length of the perimeter as measured by tracing the length of
    the interface between contacting and non-contacting points.
    """

    return np.sum(np.logical_not(c) * coordination(c, stencil=stencil))


def outer_perimeter(c, stencil=nn_stencil):
    """
    Return a map where surface points on the outer perimeter are marked.
    """

    return np.logical_and(np.logical_not(c),
                          coordination(c, stencil=stencil) > 0)


def inner_perimeter(c, stencil=nn_stencil):
    """
    Return a map where surface points on the inner perimeter are marked.
    """

    return np.logical_and(c, coordination(c, stencil=stencil) < len(stencil))


def patch_areas(patch_ids):
    """
    Return a list containing patch areas
    """

    return np.bincount(patch_ids.reshape((-1,)))[1:]


def patch_forces(patch_ids, forces):
    """
    Return a list containing patch forces
    """

    return np.bincount(patch_ids.reshape((-1,)),
                       weights=forces.reshape((-1,)))[1:]


def patch_perimeters(patch_ids, perim):
    """
    Return a list containing patch perimeters
    """

    return np.bincount(patch_ids.reshape((-1,)),
                       weights=perim.reshape((-1,)))[1:]
