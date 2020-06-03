#
# Copyright 2016, 2020 Lars Pastewka
#           2019 Antoine Sanner
#
# ### MIT license
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

"""
Command line front-end for plotting 2D surfaces
"""

from argparse import ArgumentParser

import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt

from SurfaceTopography.IO import read_topography

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

surface = read_topography(arguments.filename)

###

try:
    sx, sy = surface.physical_sizes
except TypeError:
    sx, sy = surface.nb_grid_pts
nx, ny = surface.nb_grid_pts

fig = plt.figure()
ax = fig.add_subplot(111, aspect=sx/sy)
Y, X = np.meshgrid((np.arange(ny)+0.5)*sy/ny,
                   (np.arange(nx)+0.5)*sx/nx)
Z = surface[...]
if arguments.fftshift:
    Z = np.fft.fftshift(Z)
if arguments.logscale:
    mesh = ax.pcolormesh(X, Y, Z,
                         norm=colors.LogNorm(vmin=Z.min(), vmax=Z.max()),rasterized=True)
else:
    mesh = ax.pcolormesh(X, Y, Z)
plt.colorbar(mesh, ax=ax)
ax.set_xlim(0, sx)
ax.set_ylim(0, sy)
if 'unit' in surface.info:
    unit = surface.info['unit']
else:
    unit = 'a.u.'
ax.set_xlabel('Position $x$ ({})'.format(unit))
ax.set_ylabel('Position $y$ ({})'.format(unit))
plt.savefig('{}.png'.format(arguments.filename))
plt.show()
