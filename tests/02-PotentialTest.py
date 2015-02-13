#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# @file   02-PotentialTest.py
#
# @author Till Junge <till.junge@kit.edu>
#
# @date   27 Jan 2015
#
# @brief  Tests the potential classes
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

import unittest
import numpy as np

from PyPyContact.ContactMechanics import LJ93
from PyPyContact.ContactMechanics import LJ93smooth

import PyPyContact.Tools as Tools

if __name__ == "__main__":
    from lj93_ref_potential import V as LJ_ref_V, dV as LJ_ref_dV, d2V as LJ_ref_ddV
    from lj93smooth_ref_potential import V as LJs_ref_V, dV as LJs_ref_dV, d2V as LJs_ref_ddV
else:
    from .lj93_ref_potential import V as LJ_ref_V, dV as LJ_ref_dV, d2V as LJ_ref_ddV
    from .lj93smooth_ref_potential import V as LJs_ref_V, dV as LJs_ref_dV, d2V as LJs_ref_ddV


class LJTest(unittest.TestCase):
    tol = 1e-14
    def setUp(self):
        self.eps = 1+np.random.rand()
        self.sig = 3+np.random.rand()
        self.gam = (5+np.random.rand())
        self.rcut = 2.5*self.sig+np.random.rand()
        self.r = np.arange(.65, 3, .01)*self.sig

    def test_LJReference(self):
        """ compare lj93 to reference implementation
        """
        V, dV, ddV = LJ93(
            self.eps, self.sig, self.rcut).evaluate(
                self.r, pot=True, forces=True, curb=True)
        V_ref   = LJ_ref_V  (self.r, self.eps, self.sig, self.rcut)
        dV_ref  = -LJ_ref_dV (self.r, self.eps, self.sig, self.rcut)
        ddV_ref = LJ_ref_ddV(self.r, self.eps, self.sig, self.rcut)

        err_V   = ((  V-  V_ref)**2).sum()
        err_dV  = (( dV- dV_ref)**2).sum()
        err_ddV = ((ddV-ddV_ref)**2).sum()
        error   = err_V + err_dV + err_ddV
        self.assertTrue(error < self.tol)

    def test_LJsmoothReference(self):
        """ compare lj93smooth to reference implementation
        """
        smooth_pot = LJ93smooth(self.eps, self.sig, self.gam)
        rc1 = smooth_pot.r_t
        rc2 = smooth_pot.r_c
        V, dV, ddV = smooth_pot.evaluate(
                self.r, pot=True, forces=True, curb=True)
        V_ref   = LJs_ref_V  (self.r, self.eps, self.sig, rc1, rc2)
        dV_ref  = -LJs_ref_dV (self.r, self.eps, self.sig, rc1, rc2)
        ddV_ref = LJs_ref_ddV(self.r, self.eps, self.sig, rc1, rc2)

        err_V   = ((  V-  V_ref)**2).sum()
        err_dV  = (( dV- dV_ref)**2).sum()
        err_ddV = ((ddV-ddV_ref)**2).sum()
        error   = err_V + err_dV + err_ddV
        self.assertTrue(
            error < self.tol,
            ("Error = {}, (tol = {})\n"
             "   err_V = {}, err_dV = {}, err_ddV = {}").format(
            error, self.tol, err_V, err_dV, err_ddV))

    def test_LJsmoothSanity(self):
        """ make sure LJsmooth rejects common bad input
        """
        self.assertRaises(LJ93smooth.PotentialError, LJ93smooth,
                          self.eps, self.sig, -self.gam)

    def test_LJ_gradient(self):
        pot = LJ93(self.eps, self.sig, self.rcut)
        x = np.random.random(3)-.5+self.sig
        V, dV, ddV = pot.evaluate(x, forces=True)
        f = V.sum()
        g = -dV

        delta = self.sig/1e5
        approx_g = Tools.evaluate_gradient(
            lambda x: pot.evaluate(x)[0].sum(),
            x, delta)
        tol = 1e-8
        error = Tools.mean_err(g, approx_g)
        msg = []
        msg.append("f = {}".format(f))
        msg.append("g = {}".format(g))
        msg.append('approx = {}'.format(approx_g))
        msg.append("error = {}".format(error))
        msg.append("tol = {}".format(tol))
        self.assertTrue(error < tol, ", ".join(msg))

    def test_LJsmooth_gradient(self):
        pot = LJ93smooth(self.eps, self.sig, self.gam)
        x = np.random.random(3)-.5+self.sig
        V, dV, ddV = pot.evaluate(x, forces=True)
        f = V.sum()
        g = -dV

        delta = self.sig/1e5
        approx_g = Tools.evaluate_gradient(
            lambda x: pot.evaluate(x)[0].sum(),
            x, delta)
        tol = 1e-8
        error = Tools.mean_err(g, approx_g)
        msg = []
        msg.append("f = {}".format(f))
        msg.append("g = {}".format(g))
        msg.append('approx = {}'.format(approx_g))
        msg.append("error = {}".format(error))
        msg.append("tol = {}".format(tol))
        self.assertTrue(error < tol, ", ".join(msg))


    def test_single_point_eval(self):
        pot = LJ93(self.eps, self.sig, self.gam)
        r_m = pot.r_min
        curb = pot.evaluate(r_m, pot=False, forces=False, curb=True)[2]
    def test_ad_hoc(self):
        ## 'Potential 'lj9-3smooth', ε = 1.7294663266397667,
        ## σ = 3.253732668164946, γ = 5.845648523044794, r_t = r_min' failed.
        ## Please check whether the inputs make sense
        eps = 1.7294663266397667
        sig = 3.253732668164946
        gam = 5.845648523044794
        smooth_pot = LJ93smooth(eps, sig, gam)

        rc1 = smooth_pot.r_t
        rc2 = smooth_pot.r_c
        V, dV, ddV = smooth_pot.evaluate(
                self.r, pot=True, forces=True, curb=True)
        V_ref   = LJs_ref_V  (self.r, eps, sig, rc1, rc2)
        dV_ref  = -LJs_ref_dV (self.r, eps, sig, rc1, rc2)
        ddV_ref = LJs_ref_ddV(self.r, eps, sig, rc1, rc2)

        err_V   = ((  V-  V_ref)**2).sum()
        err_dV  = (( dV- dV_ref)**2).sum()
        err_ddV = ((ddV-ddV_ref)**2).sum()
        error   = err_V + err_dV + err_ddV
        self.assertTrue(
            error < self.tol,
            ("Error = {}, (tol = {})\n"
             "   err_V = {}, err_dV = {}, err_ddV = {}").format(
            error, self.tol, err_V, err_dV, err_ddV))

if __name__ == '__main__':
    unittest.main()
