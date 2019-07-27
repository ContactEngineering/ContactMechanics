#
# Copyright 2018-2019 Antoine Sanner
#           2016, 2019 Lars Pastewka
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
Exponential attraction.
"""

from . import Potential
import numpy as np
from NuMPI import MPI

class Exponential(Potential):
    """ V(g) = -gamma0*e^(-g(r)/rho)
    """

    name = "adh"

    def __init__(self, gamma0,rho,r_cut=float('inf'), communicator=MPI.COMM_WORLD):
        """
        Keyword Arguments:
        gamma0 -- surface energy at perfect contact
        rho   -- attenuation length
        """
        self.rho = rho
        self.gam = gamma0
        Potential.__init__(self,r_cut,communicator=communicator)


    def __repr__(self, ):
        return ("Potential '{0.name}': eps = {0.eps}, sig = {0.sig},"
                "r_c = {1}").format(
                    self, self.r_c if self.has_cutoff else 'None')

    def __getstate__(self):
        state = super().__getstate__(), self.rho, self.gam
        return state

    def __setstate__(self, state):
        superstate, self.rho, self.gam = state
        super().__setstate__(superstate)

    @property
    def r_min(self):
        return None

    @property
    def r_infl(self):
        return None

    def naive_pot(self, r,pot=True,forces=False,curb=False):
        """ Evaluates the potential and its derivatives without cutoffs or
            offsets. These have been collected in a single method to reuse the
            computated LJ terms for efficiency
            V(g) = -gamma0*e^(-g(r)/rho)
            V'(g) = (gamma0/rho)*e^(-g(r)/rho)
            V''(g) = -(gamma0/r_ho^2)*e^(-g(r)/rho)

            Keyword Arguments:
            r      -- array of distances
            pot    -- (default True) if true, returns potential energy
            forces -- (default False) if true, returns forces
            curb   -- (default False) if true, returns second derivative
        """
        # pylint: disable=bad-whitespace
        # pylint: disable=invalid-name
        g = -r/self.rho

        # Use exponential only for r > 0
        m = g < 0.0
        if np.isscalar(r):
            if m:
                V = -self.gam*np.exp(g)
                dV = V/self.rho
                ddV = V/self.rho**2
            else:
                V = -self.gam*(1+g+0.5*g**2)
                dV = -self.gam/self.rho*(1+g)
                ddV = -self.gam/self.rho**2
        else:
            V = np.zeros_like(g)
            dV = np.zeros_like(g)
            ddV = np.zeros_like(g)
            V[m] = -self.gam*np.exp(g[m])
            dV[m] = V[m]/self.rho
            ddV[m] = V[m]/self.rho**2

            # Quadratic function for r < 0. This avoids numerical overflow at small r.
            m = np.logical_not(m)
            V[m] = -self.gam*(1+g[m]+0.5*g[m]**2)
            dV[m] = -self.gam/self.rho*(1+g[m])
            ddV[m] = -self.gam/self.rho**2

        return V, dV, ddV
