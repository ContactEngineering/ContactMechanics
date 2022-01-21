#
# Copyright 2022 Lars Pastewka
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

import numpy as np


def scan_with_rigid_sphere(topography, radius):
    """
    Scan a topography with a rigid, spherical tip. This emulated scanning of
    a physical topography in an scanning probe instrument.

    Paramaters
    ----------
    topography : :obj:`SurfaceTopography.UniformLineScan` or :obj:`SurfaceTopography.NonuniformLineScan`
        Topography to be scanned.
    radius : float
        Tip radius.

    Returns
    -------
    scanned_heights : np.ndarray
         Scannned heights of topography on the same grid as the topography.
    """
    if topography.dim != 1:
        raise ValueError('Only one-dimensional scans are supported at present.')

    positions, heights = topography.positions_and_heights()
    scanned_heights = []
    for x in positions:
        left = np.searchsorted(positions, x - radius)
        right = np.searchsorted(positions, x + radius)

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.plot(positions[left:right], heights[left:right], 'k-')
        # plt.plot(positions[left:right], - np.sqrt(radius ** 2 - (positions[left:right] - x) ** 2), 'k--')
        # plt.plot(positions[left:right],
        #          - np.sqrt(radius ** 2 - (positions[left:right] - x) ** 2) - heights[left:right], 'r-')
        # plt.show()

        scanned_heights += [
            np.max(heights[left:right] + np.sqrt(radius ** 2 - (positions[left:right] - x) ** 2) - radius)]

    return np.array(scanned_heights)
