#
# Copyright 2016-2017 Lars Pastewka
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
Harmonic potential for wall interaction
"""

from . import Potential


class HarmonicPotential(Potential):
    """ Repulsive harmonic potential.

        Harmonic potential:
        V (r) = 1/2 k r**2 for r < 0
    """

    name = "lj9-3"

    def __init__(self, spring_constant):
        """
        Keyword Arguments:
        spring_constant -- Spring constant k
        """
        self.spring_constant = spring_constant
        Potential.__init__(self, 0)

    def __repr__(self, ):
        return ("Potential '{0.name}': k = {0.spring_constant}").format(self)

    @property
    def r_min(self):
        """convenience function returning the location of the energy minimum"""
        return 0

    @property
    def r_infl(self):
        """convenience function returning the location of the potential's
        inflection point (if applicable)
        """
        return None

    def naive_pot(self, r, pot=True, forces=False, curb=False):
        """ Evaluates the potential and its derivatives without cutoffs or
            offsets.
        """
        # pylint: disable=bad-whitespace
        # pylint: disable=invalid-name
        V = dV = ddV = None
        if pot:
            V = 0.5*self.spring_constant*r**2
        if forces:
            # Forces are the negative gradient
            dV = -self.spring_constant*r
        if curb:
            ddV = self.spring_constant
        return (V, dV, ddV)
