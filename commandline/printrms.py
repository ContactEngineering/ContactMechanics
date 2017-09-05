#! /usr/bin/env python3
"""
@file   printrms.py

@author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>

@date   04 September 2017

@brief  Command line front-end for printing statistical properties to screen

@section LICENCE

 Copyright (C) 2017 Lars Pastewka

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

from argparse import ArgumentParser, ArgumentTypeError

import numpy as np
import matplotlib.pyplot as plt

from PyCo.Surface.FromFile import read, detect_format

###

parser = ArgumentParser(description='Print statistical properties to screen.')
parser.add_argument('filename', metavar='FILENAME', help='name of map file')
arguments = parser.parse_args()

###

surfaces = read(arguments.filename, format=detect_format(arguments.filename))

if type(surfaces) is not list:
    surfaces = [surfaces]

###

for surface in surfaces:
    try:
        print('---', surface.info['data'], '---')
    except:
        print('---')
    print('RMS height =', surface.compute_rms_height(), surface.unit)
    print('RMS slope = ', surface.compute_rms_slope())
    print('RMS curvature =', surface.compute_rms_curvature(),
          '{}^-1'.format(surface.unit))
