#
# Copyright 2017 Lars Pastewka
#           2016 Till Junge
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
Defines the base class for contact description
"""

import numpy as np


class Interaction(object):
    "base class for all interactions, e.g. interatomic potentials"
    # pylint: disable=too-few-public-methods
    pass


class HardWall(Interaction):
    "base class for non-smooth contact mechanics"
    # pylint: disable=too-few-public-methods
    def __init__(self):
        self.penetration = None

    def compute(self, gap, tol=0.):
        """
        Keyword Arguments:
        gap -- array containing the point-wise gap values
        tol -- tolerance for determining whether the gap is closed
        """
        self.penetration = np.where(gap < tol, -gap, 0)


class SoftWall(Interaction):
    "base class for smooth contact mechanics"
    def __init__(self):
        self.energy = None
        self.force = None

    def compute(self, gap, pot=True, forces=False, area_scale=1.):
        """
        computes and stores the interaction energy and/or forces based on the
        as fuction of the gap
        Parameters:
        gap        -- array containing the point-wise gap values
        pot        -- (default True) whether the energy should be evaluated
        forces     -- (default False) whether the forces should be evaluated
        area_scale -- (default 1.) scale by this. (Interaction quantities are
                      supposed to be expressed per unit area, so systems need
                      to be able to scale their response for their resolution))
        """
        energy, self.force = self.evaluate(
            gap, pot, forces, area_scale)
        self.energy = energy.sum()

    def evaluate(self, gap, pot=True, forces=False, area_scale=1.):
        """
        computes and returns the interaction energy and/or forces based on the
        as fuction of the gap
        Parameters:
        gap        -- array containing the point-wise gap values
        pot        -- (default True) whether the energy should be evaluated
        forces     -- (default False) whether the forces should be evaluated
        area_scale -- (default 1.) scale by this. (Interaction quantities are
                      supposed to be expressed per unit area, so systems need
                      to be able to scale their response for their resolution))
        """
        raise NotImplementedError()
