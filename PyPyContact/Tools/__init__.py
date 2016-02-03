#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   __init__.py

@author Till Junge <till.junge@kit.edu>

@date   27 Jan 2015

@brief  Helper tools for PyPyContact

@section LICENCE

 Copyright (C) 2015 Till Junge

PyPyContact is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation, either version 3, or (at
your option) any later version.

PyPyContact is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with GNU Emacs; see the file COPYING. If not, write to the
Free Software Foundation, Inc., 59 Temple Place - Suite 330,
Boston, MA 02111-1307, USA.
"""

from .common import compare_containers, evaluate_gradient, mean_err
from .common import compute_wavevectors, fftn, ifftn
from .common import shift_and_tilt, shift_and_tilt_approx
from .common import compute_rms_slope
from .DistributedComputation import BaseResultManager, BaseWorker
from .Logger import Logger
from . import Optimisation
