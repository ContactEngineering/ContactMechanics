#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   __init__.py

@author Till Junge <till.junge@kit.edu>

@date   17 Sep 2015

@brief  optimisation helpers in PyPyContact

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

from .common import ReachedTolerance, ReachedMaxiter, FailedIterate
from .common import ReachedMaxiterWarning
from .common import intersection_confidence_region
from .common import dogleg
from .common import steihaug_toint
from .common import modified_cholesky
from .common import first_wolfe_condition
from .common import second_wolfe_condition
from .common import line_search
from .AugmentedLagrangian import augmented_lagrangian
from .NewtonLineSearch import newton_linesearch
from .NewtonConfidenceRegion import newton_confidence_region
