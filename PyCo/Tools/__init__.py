#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   __init__.py

@author Till Junge <till.junge@kit.edu>

@date   27 Jan 2015

@brief  Helper tools for PyCo

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

from .common import (compare_containers, evaluate_gradient, mean_err,
                     compute_wavevectors, fftn, ifftn, shift_and_tilt,
                     shift_and_tilt_approx, shift_and_tilt_from_slope,
                     compute_rms_height, compute_derivative, compute_rms_slope,
                     compute_rms_curvature, get_q_from_lambda,
                     power_spectrum_1D, power_spectrum_2D,
                     compute_tilt_from_height, compute_tilt_from_slope,
                     compute_tilt_and_curvature, _get_size)
from .DistributedComputation import BaseResultManager, BaseWorker
from .Logger import Logger
from . import Optimisation
from . import fftext

compute_slope = compute_derivative