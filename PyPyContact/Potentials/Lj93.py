#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# @file   Lj93.py
#
# @author Till Junge <till.junge@kit.edu>
#
# @date   22 Jan 2015
#
# @brief  9-3 Lennard-Jones potential for wall interactions
#
# @section LICENCE
#
#  Copyright (C) 2015 Till Junge
#
# PyPyContact is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation, either version 3, or (at
# your option) any later version.
#
# PyPyContact is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GNU Emacs; see the file COPYING. If not, write to the
# Free Software Foundation, Inc., 59 Temple Place - Suite 330,
# Boston, MA 02111-1307, USA.
#
import Potential

class LJ93(Potential.Potential):
    """ 9-3 Lennard-Jones potential with optional cutoff radius.

        9-3 Lennard-Jones potential:
        V_l (r) = ε[ 2/15 (σ/r)**9 - (σ/r)**3]

        When used with a cutoff radius, the potential is shifted in orden to
        guarantee continuity of forces

        V_lc (r) = V_l (r) - V_l (r_c)
    """
    def __init__(self, epsilon, sigma, r_cut=float('inf')):
        """
        Keyword Arguments:
        epsilon -- Lennard-Jones potential well
        sigma   -- Lennard-Jones distance parameter
        r_cut   -- (default infinity) optional cutoff radius
        """
        self.eps = epsilon
        self.sig = sigma
        self.r_c = r_cut
        self.has_cutoff = not math.isinf(r_c)

    def naive_V(self, r, pot=True, forces=False, curb=False):
        """ Evaluates the potential without cutoffs or offsets
        Keyword Arguments:
        r      -- array of distances
        forces -- (default False) if true, returns forces as well
        """
        sig_r3 = (self.sig/r)**3
        sig_r9 = sig_r3**3
        V =  self.eps*(2./15*sig_r9 - sig_r3)
        if forces: pass



if __name__ == "__main__":
    from sympy import Symbol, pprint
    import sympy
    sig = Symbol('sig')
    eps = Symbol('eps')
    r = Symbol('r')
    lj = eps*(2/15*(sig/r)**9 - (sig/r)**3)
    dlj = sympy.diff(lj, r)
    ddlj = sympy.diff(dlj, r)
    pprint(  lj)
    pprint( dlj)
    pprint(ddlj)

