#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   __init__.py

@author Till Junge <till.junge@kit.edu>

@date   21 Jan 2015

@brief  Defines all interaction modes used in PyCo

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
from .Interactions import Interaction, HardWall, SoftWall
from .Potentials import Potential, SmoothPotential, MinimisationPotential
from .Potentials import SimpleSmoothPotential
from .Lj93 import LJ93, LJ93smooth, LJ93smoothMin, LJ93SimpleSmooth
from .VdW82 import VDW82, VDW82smooth, VDW82smoothMin, VDW82SimpleSmooth
