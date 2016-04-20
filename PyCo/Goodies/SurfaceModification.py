#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   SurfaceModification.py

@author Till Junge <till.junge@kit.edu>

@date   18 Jun 2015

@brief  Helper functions for the generation of random fractal surfaces

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
from scipy.optimize import brentq

from .SurfaceAnalysis import CharacterisePeriodicSurface
from .SurfaceGeneration import RandomSurfaceExact
from ..Tools.common import compute_rms_slope, compute_rms_curvature, fftn


class ModifyExistingPeriodicSurface(RandomSurfaceExact):
    """ Metasurface based on existing surface"""
    def __init__(self, surface):
        """
        Generates a surface with an Gaussian amplitude distribution
        Keyword Arguments:
        surface -- Surface to be modified.
        """
        self.surface = surface
        surf_char = CharacterisePeriodicSurface(self.surface)
        hurst = surf_char.estimate_hurst()
        rms_height = self.surface.compute_rms_height()
        super().__init__(surface.resolution, surface.size, hurst, rms_height,
                         seed=None, lambda_max=None)

    def generate_phases(self):
        return 0

    def generate_amplitudes(self):
        area = np.prod(self.size)
        self.coeffs = fftn(self.surface.profile(), area)


def estimate_short_cutoff(surface, rms_slope=None, rms_curvature=None,
                          return_bounds=False):
    surf = ModifyExistingPeriodicSurface(surface)
    cutoff1 = np.mean(surf.size)
    cutoff2 = np.mean([x/y for x, y in zip(surf.size, surf.resolution)])/2
    initial_slope = compute_rms_slope(surf.get_surface())
    initial_curvature = compute_rms_curvature(surf.get_surface())
    if rms_slope is not None:
        if rms_curvature is not None:
            raise ValueError('Please specify either target rms slope or target '
                             'rms curvature, not both.')
        slope1 = compute_rms_slope(surf.get_surface(lambda_min=cutoff1))
        slope2 = compute_rms_slope(surf.get_surface(lambda_min=cutoff2))
        if rms_slope < slope1 or rms_slope > slope2:
            if return_bounds:
                if rms_slope < slope1:
                    return cutoff1
                else:
                    return cutoff2
            else:
                raise ValueError('Target slope value (={}) must lie between '
                                 'slopes for largest (={}) and smallest (={}) '
                                 'small wavelength cutoff.' \
                                 .format(rms_slope, slope1, slope2))
        cutoff = brentq(lambda cutoff: compute_rms_slope(
            surf.get_surface(lambda_min=cutoff))-rms_slope, cutoff1, cutoff2)
    elif rms_curvature is not None:
        curvature1 = compute_rms_curvature(surf.get_surface(lambda_min=cutoff1))
        curvature2 = compute_rms_curvature(surf.get_surface(lambda_min=cutoff2))
        if rms_curvature < curvature1 or rms_curvature > curvature2:
            if return_bounds:
                if rms_curvature < curvature1:
                    return cutoff1
                else:
                    return cutoff2
            else:
                raise ValueError('Target curvature value (={}) must lie '
                                 'between curvatures for largest (={}) and '
                                 'smallest (={}) small wavelength cutoff.' \
                                 .format(rms_curvature, curvature1, curvature2))
        cutoff = brentq(lambda cutoff: compute_rms_curvature(
            surf.get_surface(lambda_min=cutoff))-rms_curvature, cutoff1,
            cutoff2)
    else:
        raise ValueError('Please specify either target rms slope or target rms '
                         'curvature.')
    return cutoff
