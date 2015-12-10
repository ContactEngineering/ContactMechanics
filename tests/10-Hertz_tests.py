#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   10-Hertz_tests.py

@author Till Junge <till.junge@kit.edu>

@date   05 Oct 2015

@brief  Tests adhesion-free systems for accuracy and compares performance

@section LICENCE

 Copyright (C) 2015 Till Junge

PyPyContact is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation, either version 3, or (at
your option) any later version.

PyPyContact is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with GNU Emacs; see the file COPYING. If not, write to the
Free Software Foundation, Inc., 59 Temple Place - Suite 330,
Boston, MA 02111-1307, USA.
"""

import unittest
import numpy as np
import time
import math
from PyPyContact.ContactMechanics import HardWall
from PyPyContact.SolidMechanics import PeriodicFFTElasticHalfSpace
from PyPyContact.Surface import Sphere
from PyPyContact.System import SystemFactory
from PyPyContact.Tools.Logger import screen

# -----------------------------------------------------------------------------
# Taken from matscipy
def radius_and_pressure(N, R, Es):
    """
    Given normal load, sphere radius and contact modulus compute contact radius
    and peak pressure.

    Parameters
    ----------
    N : float
        Normal force.
    R : float
        Sphere radius.
    Es : float
        Contact modulus: Es = E/(1-nu**2) with Young's modulus E and Poisson
        number nu.

    Returns
    -------
    a : float
        Contact radius.
    p0 : float
        Maximum pressure inside the contacting area (right under the apex).
    """

    a = R*(3./4*( N/(Es*R**2) ))**(1./3)
    p0 = 3*N/(2*math.pi*a*a)
    
    return a, p0

# -----------------------------------------------------------------------------
# Taken from matscipy
def surface_stress(r, nu=0.5):
    """
    Given distance from the center of the sphere, contact radius and Poisson
    number contact, compute the stress at the surface.

    Parameters
    ----------
    r : array_like
        Array of distance (from the center of the sphere in units of contact
        radius a).
    nu : float
        Poisson number.

    Returns
    -------
    pz : array
        Contact pressure (in units of maximum pressure p0).
    sr : array
        Radial stress (in units of maximum pressure p0).
    stheta : array
        Azimuthal stress (in units of maximum pressure p0).
    """

    mask0 = np.abs(r) < 1e-6
    maski = np.logical_and(r < 1.0, np.logical_not(mask0))
    masko = np.logical_and(np.logical_not(maski), np.logical_not(mask0))
    r_0 = r[mask0]
    r_i = r[maski]
    r_o = r[masko]

    # Initialize
    pz = np.zeros_like(r)
    sr = np.zeros_like(r)
    stheta = np.zeros_like(r)

    # Solution at r=0
    if mask0.sum() > 0:
        pz[mask0] = np.ones_like(r_0)
        sr[mask0] = -(1.+2*nu)/2.*np.ones_like(r_0)
        stheta[mask0] = -(1.+2*nu)/2.*np.ones_like(r_0)

    # Solution inside the contact radius
    if maski.sum() > 0:
        r_a_sq = r_i**2
        pz[maski] = np.sqrt(1-r_a_sq)
        sr[maski] = (1.-2.*nu)/(3.*r_a_sq)*(1.-(1.-r_a_sq)**(3./2))- \
            np.sqrt(1.-r_a_sq)
        stheta[maski] = -(1.-2.*nu)/(3.*r_a_sq)*(1.-(1.-r_a_sq)**(3./2))- \
            2*nu*np.sqrt(1.-r_a_sq)

    # Solution outside of the contact radius
    if masko.sum() > 0:
        r_a_sq = r_o**2
        po = (1.-2.*nu)/(3.*r_a_sq)
        sr[masko] = po
        stheta[masko] = -po

    return pz, sr, stheta

# -----------------------------------------------------------------------------
# Taken from matscipy
def surface_displacements(r):
    """
    Return the displacements at the surface due to an indenting sphere.
    See: K.L. Johnson, Contact Mechanics, p. 61

    Parameters
    ----------
    r : array_like
        Radial position normalized by contact radius a.

    Returns
    -------
    uz : array
        Normal displacements at the surface of the contact (in units of
        p0/Es * a where p0 is maximum pressure, Es contact modulus and a
        contact radius).
    """

    maski = r < 1.0
    masko = np.logical_not(maski)
    r_i = r[maski]
    r_o = r[masko]

    # Initialize
    uz = np.zeros_like(r)

    # Solution inside the contact circle
    if maski.sum() > 0:
        uz[maski] = -math.pi*(2.-r_i**2)/4.

    # Solution outside the contact circle
    if masko.sum() > 0:
        uz[masko] = (-(2.-r_o**2)*np.arcsin(1./r_o) -
            r_o*np.sqrt(1.-(1./r_o)**2))/2.

    return uz




# -----------------------------------------------------------------------------
class HertzTest(unittest.TestCase):
    def setUp(self):
        # sphere radius:
        self.r_s = 20.0
        # contact radius
        self.r_c = .2
        # peak pressure
        self.p_0 = 2.5
        # equivalent Young's modulus
        self.E_s = 102.

    def test_elastic_solution(self):
        r = np.linspace(0, self.r_s, 6)/self.r_c
        u = surface_displacements(r) / (self.p_0/self.E_s*self.r_c)
        sig = surface_stress(r)[0]/self.p_0

    def test_constrained_conjugate_gradients(self):
        nx = 512
        sx = 15.0
        disp0 = 0.1
        substrate = PeriodicFFTElasticHalfSpace((nx, nx), self.E_s)
        interaction = HardWall()
        surface = Sphere(self.r_s, (nx, nx), (sx, sx))
        system = SystemFactory(substrate, interaction, surface)

        disp, forces = system.minimize_proxy(disp0, logger=screen)

        print(forces.sum(), forces[forces>0].sum(), forces[forces<0].sum())
        normal_force = -forces.sum()
        a, p0 = radius_and_pressure(normal_force, self.r_s, self.E_s)
        print(a, p0, a/self.r_s)

        import matplotlib.pyplot as plt
        x = (np.arange(nx)-nx/2)*sx/nx
        plt.subplot(2,1,1)
        plt.plot(x, disp[:,nx/2])
        plt.plot(x, np.sqrt(self.r_s**2-x**2)-(self.r_s-disp0))
        plt.subplot(2,1,2)
        plt.plot(x, -forces[:,nx/2]*(nx/sx)**2)
        plt.plot(x, p0*np.sqrt(1-(x/a)**2))
        plt.show()


if __name__ == '__main__':
    unittest.main()