#
# Copyright 2016, 2018, 2020 Lars Pastewka
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
Command line front-end for plotting the height-height autocorrelation
function of a 2D map
"""

from argparse import ArgumentParser, ArgumentTypeError

import matplotlib.pyplot as plt

from SurfaceTopography import open_topography, DetrendedTopography
from PyCo.Tools import autocorrelation_2D

###

parser = ArgumentParser(description='Plot the power-spectral density of a 2D '
                                    'map (pressure, gap, etc.).')
parser.add_argument('filename', metavar='FILENAME', help='name of map file')
parser.add_argument('--nbins', dest='nbins', type=int, default=None,
                    help='use NBINS bins for radial average',
                    metavar='NBINS')
parser.add_argument('--detrend_mode', dest='detrend_mode', type=str,
                    default='center',
                    help='detrend surface; posibilities are '
                    'DETREND=center|height|slope|curvature',
                    metavar='DETREND')
arguments = parser.parse_args()

###

surface = DetrendedTopography(open_topography(arguments.filename),
                              detrend_mode=arguments.detrend_mode)
unit = surface.unit
nx, ny = surface.shape
nbins = arguments.nbins
if nbins is None:
    nbins = (nx+ny)/2

###

r, A, A_xy = autocorrelation_2D(surface, nbins=nbins, return_map=True)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.loglog(r, A, 'ko')
ax.set_xlabel('Distance $r$ ({})'.format(unit))
ax.set_ylabel(r'Autocorrelation $\langle h(0)h(r)\rangle$ ({}$^2$)'.format(unit))
#plt.pcolormesh(A_xy)
#plt.colorbar()
plt.show()
