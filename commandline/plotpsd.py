#! /usr/bin/env python3
"""
@file   plotpsd.py

@author Lars Pastewka <lars.pastewka@kit.edu>

@date   20 Oct 2016

@brief  Command line front-end for plotting the power-spectral density of a
        2D map

@section LICENCE

 Copyright (C) 2016 Lars Pastewka

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

import matplotlib.pyplot as plt

from PyCo.Surface import read
from PyCo.Tools import power_spectrum_2D

###

parser = ArgumentParser(description='Plot the power-spectral density of a 2D '
                                    'map (pressure, gap, etc.).')
parser.add_argument('filename', metavar='FILENAME', help='name of map file')
parser.add_argument('--nbins', dest='nbins', type=int, default=None,
                    help='use NBINS bins for radial average',
                    metavar='NBINS')
arguments = parser.parse_args()

###

surface = read(arguments.filename)
unit = surface.unit
nx, ny = surface.shape
nbins = arguments.nbins
if nbins is None:
    nbins = (nx+ny)/2

###

q, C = power_spectrum_2D(surface, nbins=nbins)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.loglog(q, C, 'ko')
ax.set_xlabel('Wavevector $q$ ({}$^{{-1}}$)'.format(unit))
ax.set_ylabel(r'Power-spectral density $C^{{\rm iso}}$ ({}$^4$)'.format(unit))
plt.show()
