#
# Copyright 2018-2019 Antoine Sanner
#           2019 Lars Pastewka
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
Tests adhesion-free systems for accuracy and compares performance
"""

try:
    import unittest
    import numpy as np
    import time
    import math
    import PyCo.ReferenceSolutions.DMT as DMT
    import PyCo.ReferenceSolutions.JKR as JKR
    import PyCo.ReferenceSolutions.MaugisDugdale as MD
    from scipy.optimize import minimize_scalar
    from PyCo.ContactMechanics import ExpPotential
    from PyCo.SolidMechanics import (FreeFFTElasticHalfSpace,
                                     PeriodicFFTElasticHalfSpace)
    from PyCo.Topography import make_sphere
    from PyCo.System import make_system, SmoothContactSystem
except ImportError as err:
    import sys
    print(err)
    sys.exit(-1)

from NuMPI import MPI
import pytest
pytestmark = pytest.mark.skipif(MPI.COMM_WORLD.Get_size()> 1,
        reason="tests only serial funcionalities, please execute with pytest")
# -----------------------------------------------------------------------------
#@unittest.skip("Skip because test is slow")
class AdhesionTest(unittest.TestCase):
    def setUp(self):
        # sphere radius:
        self.r_s = 10.0
        # contact radius
        self.r_c = .2
        # peak pressure
        self.p_0 = 2.5
        # equivalent Young's modulus
        self.E_s = 102.
        # work of adhesion
        self.w = 1.0
        # tolerance for optimizer
        self.tol = 1e-12
        # tolerance for contact area
        self.gap_tol = 1e-6

    def test_hard_wall_LBFGS(self):
        nx, ny = 128, 128
        sx = 21.0

        for ran in [0.05, 0.3]:
            substrate = FreeFFTElasticHalfSpace((nx, ny), self.E_s, (sx, sx))
            interaction = ExpPotential(self.w, ran)# , 0.13)
            surface = make_sphere(self.r_s, (nx, ny), (sx, sx), standoff=float('inf'))
            ext_surface = make_sphere(self.r_s, (2*nx, 2*ny), (2*sx, 2*sx),
                                 centre=(sx/2, sx/2), standoff=float('inf'))
            system = SmoothContactSystem(substrate, interaction, surface)

            disp0 = np.linspace(-self.r_s/100, self.r_s/50, 11)
            normal_force = []
            area = []
            for _disp0 in disp0:
                result = system.minimize_proxy(_disp0,
                                               lbounds="auto",#ext_surface.heights() + _disp0,
                                               tol=self.tol)
                u = result.x
                u.shape = ext_surface.nb_grid_pts
                f = substrate.evaluate_force(u)
                converged = result.success
                self.assertTrue(converged)

                gap = system.compute_gap(u, _disp0)

                normal_force += [-f.sum()]
                area += [(gap<self.gap_tol).sum()*system.area_per_pt]

            normal_force = np.array(normal_force)
            area = np.array(area)

            opt = minimize_scalar(lambda x: ((MD.load_and_displacement(np.sqrt(area/np.pi), self.r_s, self.E_s, self.w, x)[0]-normal_force)**2).sum(),
                                  bracket=(0.1*self.w/ran, 2*self.w/ran))
            cohesive_stress = opt.x

            residual = np.sqrt(((MD.load_and_displacement(np.sqrt(area/np.pi), self.r_s, self.E_s, self.w, cohesive_stress)[0]-normal_force)**2).mean())
            self.assertTrue(residual < 1, msg = "residual = {} >=01".format(residual))

            if False:
                import matplotlib.pyplot as plt

                fig,ax = plt.subplots()

                ax.plot(area, MD.load_and_displacement(np.sqrt(area/np.pi), self.r_s, self.E_s, self.w, cohesive_stress)[0], label="analytical")
                ax.plot(area, normal_force, label="numerical")


                ax.set_xlabel("area")
                ax.set_ylabel("normal_force")
                ax.grid(True)
                ax.legend()

                fig.tight_layout()
                plt.show(block=True)

if __name__ == '__main__':
    unittest.main()
