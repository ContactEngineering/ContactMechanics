#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# @file   Lj93smooth.py
#
# @author Till Junge <till.junge@kit.edu>
#
# @date   21 Jan 2015
#
# @brief  9-3 Lennard-Jones potential with forces splined to zero from
# the minimum of the potential using a cubic spline.
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


class Lj93smooth(Potential):
    """ 9-3 Lennard-Jones potential with forces splined to zero from
        the minimum of the potential using a fourth order spline. The 9-3
        Lennard-Jones interaction potential is often used to model the interac-
        tion between a continuous solid wall and the atoms/molecules of a li-
        quid.

        9-3 Lennard-Jones potential:
        V_l (r) = ε[ 2/15 (σ/r)**9 - (σ/r)**3]

        decoupled 9-3:
        V_lγ (r) = V_l(r) - V_l(r_t) + γ

        spline: (Δr = r-r_t)
        V_s(Δr) = C0 - C1*Δr
                  - 1/2*C2*Δr**2
                  - 1/3*C3*Δr**3
                  - 1/4*C4*Δr**4

        The formulation allows to choose the contact stiffness (the original
        repulsive LJ zone) independently from the work of adhesion (the energy
        well). By default, the work of adhesion equals epsilon and the transi-
        tion point r_t between LJ and spline is at the minimumm, however they
        can be chosen freely.

        The spline is chosen to guarantee continuity of the second derivative of
        the potential, leading to the following conditions:
        V_s' (0) = 0
        V_s''(0) = V_lγ''(r_t)
        V_s(r_c) = V_s'(r_c) = V_s''(r_c) =0
    """
    def __init__(self, epsilon, sigma, gamma=None, r_t=None):
        """
        Keyword Arguments:
        epsilon -- Lennard-Jones potential well
        sigma   -- Lennard-Jones distance parameter
        gamma   -- (default None) Work of adhesion, defaults to epsilon
        r_t     -- (default None) transition point, defaults to r_min
        """
        
