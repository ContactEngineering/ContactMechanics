#
# Copyright 2016, 2018, 2020 Lars Pastewka
#           2018, 2020 Antoine Sanner
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
Tests adhesion-free periodic calculations
"""

import numpy as np

from ContactMechanics import make_system, PeriodicFFTElasticHalfSpace
from SurfaceTopography.Generation import fourier_synthesis


def test_constrained_conjugate_gradients():
    nb_grid_pts = (512, 512)
    physical_sizes = (1., 1.)
    hurst = 0.8
    rms_slope = 0.1
    modulus = 1

    np.random.seed(999)
    topography = fourier_synthesis(nb_grid_pts, physical_sizes, hurst,
                                   rms_slope=rms_slope)

    substrate = PeriodicFFTElasticHalfSpace(nb_grid_pts, modulus,
                                            physical_sizes)
    system = make_system(substrate, topography)
    system.minimize_proxy(offset=0.1)
