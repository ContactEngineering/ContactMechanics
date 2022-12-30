#
# Copyright 2017, 2020 Lars Pastewka
#           2019-2020 Antoine Sanner
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
implements plastic mapping algorithms for contact systems
"""

import numpy as np

import SurfaceTopography
from .FFTElasticHalfSpace import ElasticSubstrate
from .Systems import NonSmoothContactSystem


class PlasticNonSmoothContactSystem(NonSmoothContactSystem):
    """
    This system implements a simple penetration hardness model.
    """

    @staticmethod
    def handles(substrate_type, surface_type, is_domain_decomposed):
        """
        determines whether this class can handle the proposed system
        composition
        Keyword Arguments:
        substrate_type   -- instance of ElasticSubstrate subclass
        surface_type     --
        """
        is_ok = True
        # any type of substrate formulation should do
        is_ok &= issubclass(substrate_type, ElasticSubstrate)

        # any surface should do
        is_ok &= issubclass(surface_type,
                            SurfaceTopography.PlasticTopography)
        return is_ok

    def minimize_proxy(self, **kwargs):
        """
        """
        # Need to convert hardness into force units because the solvers operate
        # internally with forces, not pressures.
        hardness = self.surface.hardness * self.surface.area_per_pt
        opt = super().minimize_proxy(hardness=hardness, **kwargs)
        if opt.success:
            self.surface.plastic_displ += np.where(opt.plastic, self.compute_gap(self.disp, self.offset), 0.)
        return opt
