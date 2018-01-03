#!/usr/bin/env python3
#*- coding:utf-8 -*-
"""
@file   __init__.py

@author Till Junge <till.junge@kit.edu>

@date   27 Jan 2015

@brief  Helper tools for PyCo

@section LICENSE

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

from .common import (autocorrelation_2D, compare_containers, evaluate_gradient,
					 mean_err, compute_wavevectors, fftn, ifftn, shift_and_tilt,
                     shift_and_tilt_approx, shift_and_tilt_from_slope,
                     compute_rms_height, compute_derivative, compute_rms_slope,
                     compute_rms_curvature, get_q_from_lambda,
                     power_spectrum_1D, power_spectrum_2D,
                     compute_tilt_from_height, compute_tilt_from_slope,
                     compute_tilt_and_curvature, _get_size)
from .ContactAreaAnalysis import (assign_patch_numbers, assign_segment_numbers,
                                  distance_map, inner_perimeter,
                                  outer_perimeter, patch_areas)
from .DistributedComputation import BaseResultManager, BaseWorker
from .Logger import Logger
from . import Optimisation
from . import fftext

compute_slope = compute_derivative
