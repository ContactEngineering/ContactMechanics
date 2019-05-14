#
# Copyright 2017, 2019 Lars Pastewka
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
implements plastic mapping algorithms for contact systems
"""

import numpy as np

from .. import ContactMechanics, SolidMechanics, Topography
from .Systems import NonSmoothContactSystem

class PlasticNonSmoothContactSystem(NonSmoothContactSystem):
    """
    This system implements a simple penetration hardness model.
    """

    @staticmethod
    def handles(substrate_type, interaction_type, surface_type):
        """
        determines whether this class can handle the proposed system
        composition
        Keyword Arguments:
        substrate_type   -- instance of ElasticSubstrate subclass
        interaction_type -- instance of Interaction
        surface_type     --
        """
        is_ok = True
        # any type of substrate formulation should do
        is_ok &= issubclass(substrate_type,
                            SolidMechanics.ElasticSubstrate)
        # only hard interactions allowed
        is_ok &= issubclass(interaction_type,
                            ContactMechanics.HardWall)

        # any surface should do
        is_ok &= issubclass(surface_type,
                            Topography.PlasticTopography)
        return is_ok

    def minimize_proxy(self, **kwargs):
        """
        """
        # Need to convert hardness into force units because the solvers operate
        # internally with forces, not pressures.
        opt = super().minimize_proxy(
            hardness=self.surface.hardness*self.surface.area_per_pt,
            **kwargs
            )

        return opt
