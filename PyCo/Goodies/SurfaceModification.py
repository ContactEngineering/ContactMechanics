#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   SurfaceModification.py

@author Till Junge <till.junge@kit.edu>

@date   18 Jun 2015

@brief  Helper functions for the generation of random fractal surfaces

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
from scipy.optimize import brentq

from .SurfaceAnalysis import CharacterisePeriodicSurface
from PyCo.Topography.SurfaceGeneration import RandomSurfaceExact
from ..Tools.common import fftn


class ModifyExistingPeriodicSurface(RandomSurfaceExact):
    """ Metasurface based on existing surface"""
    def __init__(self, surface):
        """
        Generates a surface with an Gaussian amplitude distribution
        Keyword Arguments:
        surface -- Topography to be modified.
        """
        self.surface = surface
        surf_char = CharacterisePeriodicSurface(self.surface)
        hurst = surf_char.estimate_hurst()
        rms_height = self.surface.rms_height()
        super().__init__(surface.resolution, surface.size, hurst, rms_height,
                         seed=None, lambda_max=None)

    def generate_phases(self):
        return 0

    def generate_amplitudes(self):
        area = np.prod(self.size)
        self.coeffs = fftn(self.surface.array(), area)


def estimate_short_cutoff(surface, rms_slope=None, rms_curvature=None,
                          return_bounds=False):
    surf = ModifyExistingPeriodicSurface(surface)
    cutoff1 = np.mean(surf.size)
    cutoff2 = np.mean([x/y for x, y in zip(surf.size, surf.resolution)])/2
    initial_slope = rms_slope(surf.get_surface())
    initial_curvature = rms_curvature(surf.get_surface())
    if rms_slope is not None:
        if rms_curvature is not None:
            raise ValueError('Please specify either target rms slope or target '
                             'rms curvature, not both.')
        slope1 = rms_slope(surf.get_surface(lambda_min=cutoff1))
        slope2 = rms_slope(surf.get_surface(lambda_min=cutoff2))
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
        cutoff = brentq(lambda cutoff: rms_slope(
            surf.get_surface(lambda_min=cutoff)) - rms_slope, cutoff1, cutoff2)
    elif rms_curvature is not None:
        curvature1 = rms_curvature(surf.get_surface(lambda_min=cutoff1))
        curvature2 = rms_curvature(surf.get_surface(lambda_min=cutoff2))
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
        cutoff = brentq(lambda cutoff: rms_curvature(
            surf.get_surface(lambda_min=cutoff)) - rms_curvature, cutoff1,
            cutoff2)
    else:
        raise ValueError('Please specify either target rms slope or target rms '
                         'curvature.')
    return cutoff
