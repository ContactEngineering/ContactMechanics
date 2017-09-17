#! /usr/bin/env python3
"""
@file   plotmap.py

@author Lars Pastewka <lars.pastewka@kit.edu>

@date   20 Oct 2016

@brief  Command line front-end for plotting 2D surfaces

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

import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt

from PyCo.Surface.FromFile import read, detect_format

###

parser = ArgumentParser(description='Plot a 2D surface map (pressure, gap, '
                                    'etc.).')
parser.add_argument('filename', metavar='FILENAME', help='name of map file')
parser.add_argument('--logscale', help='log scaling', action='store_true',
                    default=False)
parser.add_argument('--fftshift', help='fft shift', action='store_true',
                    default=False)
arguments = parser.parse_args()

###

surface = read(arguments.filename, format=detect_format(arguments.filename))

###

try:
    sx, sy = surface.size
except TypeError:
    sx, sy = surface.shape
nx, ny = surface.shape

fig = plt.figure()
ax = fig.add_subplot(111, aspect=sx/sy)
Y, X = np.meshgrid((np.arange(ny)+0.5)*sy/ny,
                   (np.arange(nx)+0.5)*sx/nx)
Z = surface[...]
if arguments.fftshift:
    Z = np.fft.fftshift(Z)
if arguments.logscale:
    mesh = ax.pcolormesh(X, Y, Z,
                         norm=colors.LogNorm(vmin=Z.min(), vmax=Z.max()),)
else:
    mesh = ax.pcolormesh(X, Y, Z)
plt.colorbar(mesh, ax=ax)
ax.set_xlim(0, sx)
ax.set_ylim(0, sy)
if surface.unit is not None:
    unit = surface.unit
else:
    unit = 'a.u.'
ax.set_xlabel('Position $x$ ({})'.format(unit))
ax.set_ylabel('Position $y$ ({})'.format(unit))
plt.show()
