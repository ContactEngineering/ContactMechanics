#
# Copyright 2019 Lars Pastewka
#           2018-2019 Antoine Sanner
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
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
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
Tests plastic deformation
"""

try:
    import numpy as np
    import unittest
    from scipy.optimize import bisect
    from PyCo.ContactMechanics import HardWall
    from PyCo.SolidMechanics import PeriodicFFTElasticHalfSpace
    from PyCo.Topography import read, PlasticTopography
    from PyCo.System import make_system
    import os
except ImportError as err:
    import sys
    print(err)
    sys.exit(-1)

DATADIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'file_format_examples')

# -----------------------------------------------------------------------------
class PlasticTest(unittest.TestCase):
    def test_hard_wall_LBFGS(self):
        # Test that at very low hardness we converge to (almost) the bearing
        # area geometry
        surface = read(os.path.join(DATADIR, 'surface1.out'), format = "asc").topography()
        system = make_system(PeriodicFFTElasticHalfSpace(surface.resolution, 1.0),
                             HardWall(), PlasticTopography(surface, 0.0000000001))
        offset = -0.002
        result = system.minimize_proxy(offset=offset)
        c = result.jac > 0.0
        ncontact = c.sum()

        bearing_area = bisect(lambda x: (surface.heights() > x).sum() - ncontact, -0.03, 0.03)
        cba = surface.heights() > bearing_area

        self.assertTrue(np.logical_not(c == cba).sum() < 25)

if __name__ == '__main__':
    unittest.main()
