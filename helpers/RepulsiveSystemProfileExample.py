#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   RepulsiveSystemProfileExample.py

@author Till Junge <till.junge@kit.edu>

@date   11 Apr 2016

@brief  Standard program for a repulsive contact sim. The idea is to use this 
        to profile and accelerate PyCo

@section LICENCE

 Copyright (C) 2016 Till Junge

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

from PyCo.SolidMechanics import FreeFFTElasticHalfSpace
from PyCo.ContactMechanics import HardWall
from PyCo.Surface import Sphere
from PyCo.System import SystemFactory

#import matplotlib.pyplot as plt

# geometry
res = 1024
size = 65e-6
radius = 500e-6


# material
e_combo = 4e6

# system
substrate = FreeFFTElasticHalfSpace((res, res), e_combo, (size, size))
interaction = HardWall()
surface = Sphere(radius, (res, res), (size, size))
system = SystemFactory(substrate, interaction, surface)


# minimize
offset_max = 1e-6
disp = None
for offset in np.linspace(0, offset_max, 10):
    result = system.minimize_proxy(offset, disp0=disp)
    disp = system.disp

print(result.success)

