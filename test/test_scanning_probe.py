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

import pytest

from SurfaceTopography.Generation import fourier_synthesis

from ContactMechanics.ScanningProbe import scan_with_rigid_sphere


def test_scan_1d_with_rigid_sphere(plot=False):
    t = fourier_synthesis((1024,), (1,), 0.8, rms_slope=0.1)
    h = t.heights()
    scanned_h = scan_with_rigid_sphere(t, 0.3)

    # Simply check that the scanned surface is above the original one
    assert (scanned_h - h).min() > -1e-16  # a little bit of tolerance

    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        x, h = t.positions_and_heights()
        plt.plot(x, h, 'k-')
        plt.plot(x, scanned_h, 'r-')
        plt.show()


def test_scan_2d_with_rigid_sphere():
    t = fourier_synthesis((1024, 1024), (1, 1), 0.8, rms_slope=0.1)
    with pytest.raises(ValueError):
        scan_with_rigid_sphere(t, 0.1)
