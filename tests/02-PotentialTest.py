#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   02-PotentialTest.py

@author Till Junge <till.junge@kit.edu>

@date   27 Jan 2015

@brief  Tests the potential classes

@section LICENCE

Copyright 2015-2017 Till Junge, Lars Pastewka

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

try:
    import unittest
    import numpy as np

    from PyCo.ContactMechanics import LJ93
    from PyCo.ContactMechanics import LJ93smooth
    from PyCo.ContactMechanics import LJ93smoothMin
    from PyCo.ContactMechanics import LJ93SimpleSmooth

    from PyCo.ContactMechanics import VDW82
    from PyCo.ContactMechanics import VDW82smooth
    from PyCo.ContactMechanics import VDW82smoothMin
    from PyCo.ContactMechanics import VDW82SimpleSmooth

    from PyCo.ContactMechanics import ExpPotential

    import PyCo.Tools as Tools

    from .lj93_ref_potential import V as LJ_ref_V, dV as LJ_ref_dV, d2V as LJ_ref_ddV
    from .lj93smooth_ref_potential import V as LJs_ref_V, dV as LJs_ref_dV, d2V as LJs_ref_ddV
except ImportError as err:
    import sys
    print(err)
    sys.exit(-1)

class PotentialTest(unittest.TestCase):
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
        dV_ref  = LJ_ref_dV (self.r, self.eps, self.sig, self.rcut)
        ddV_ref = LJ_ref_ddV(self.r, self.eps, self.sig, self.rcut)

        err_V   = ((  V -  V_ref)**2).sum()
        err_dV  = (( dV + dV_ref)**2).sum()
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

    def test_LJsmoothMinReference(self):
        """
        compare lj93smoothmin to reference implementation (where it applies).
        """
        smooth_pot = LJ93smoothMin(self.eps, self.sig, self.gam)
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

#     def test_triplePlot(self):
#         lj_pot = LJ93(self.eps, self.sig, self.rcut)
#         gam = float(-lj_pot.evaluate(lj_pot.r_min)[0])
#         smooth_pot = LJ93smooth(self.eps, self.sig, gam)
#         min_pot = LJ93smoothMin(self.eps, self.sig, gam, lj_pot.r_min)
#         plots = (("LJ", lj_pot),
#                  ("smooth", smooth_pot),
#                  ("min", min_pot))
#         import matplotlib.pyplot as plt
#         plt.figure()
#         r = self.r
#         for name, pot in plots:
#             V, dV, ddV = pot.evaluate(r)
#             plt.plot(r, V, label=name)
#         plt.legend(loc='best')
#         plt.grid(True)
#         plt.show()

    def test_LJsmoothSanity(self):
        """ make sure LJsmooth rejects common bad input
        """
        self.assertRaises(LJ93smooth.PotentialError, LJ93smooth,
                          self.eps, self.sig, -self.gam)

    def test_LJ_gradient(self):
        pot = LJ93(self.eps, self.sig, self.rcut)
        x = np.random.random(3)-.5+self.sig
        V, f, ddV = pot.evaluate(x, forces=True)
        g = -f

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

    def test_vanDerWaalsSimple(self):
        # reproduces the graph in
        # http://dx.doi.org/10.1103/PhysRevLett.111.035502
        # originally visally compared then just regression checking
        c_sr = 2# 2.1e-78
        hamaker = 4# 68.1e-78
        vdw = VDW82(c_sr, hamaker)
        r_min = vdw.r_min
        V_min, dV_min, ddV_min = vdw.evaluate(r_min, True, True, True)
        vdws = VDW82smooth(c_sr, hamaker, r_t=2.5)
        vdwm = VDW82smoothMin(c_sr, hamaker, r_ti=1.95)
        r = np.arange(0.5, 2.01, .005)*r_min


        # import matplotlib.pyplot as plt
        # plt.figure()
        # pot = vdw.evaluate(r)[0]
        # print()
        # print("  V_min = {}".format(  V_min))
        # print(" dV_min = {}".format( dV_min))
        # print("ddV_min = {}".format(ddV_min))
        # print("transition = {}".format(vdws.r_t))
        # bot = 1.1*V_min
        # plt.plot(r, pot,label='simple')
        # plt.plot(r, vdws.evaluate(r)[0],label='smooth')
        # plt.plot(r, vdwm.evaluate(r)[0],label='minim')
        # plt.scatter(vdw.r_min, V_min, marker='+')
        # plt.ylim(bottom=bot, top=0)
        # plt.legend(loc='best')
        # plt.show()

    def test_vanDerWaals(self):
        # reproduces the graph in
        # http://dx.doi.org/10.1103/PhysRevLett.111.035502
        # originally visally compared then just regression checking
        r = np.arange(0.25, 2.01, .01)*1e-9
        c_sr = 2.1e-78
        hamaker = 68.1e-21
        pots = (("vdW", VDW82(c_sr, hamaker)),
                ("smooth", VDW82smooth(c_sr, hamaker)),
                ("min", VDW82smoothMin(c_sr, hamaker)))

        # import matplotlib.pyplot as plt
        # plt.figure()
        # bot = None
        # for name, t_h_phile in pots:
        #     pot = t_h_phile.evaluate(r)[0]
        #     if bot is None:
        #         bot = 1.1*t_h_phile.evaluate(t_h_phile.r_min)[0]
        #     plt.plot(r, pot,label=name)
        # plt.ylim(bottom=bot, top=0)
        # plt.legend(loc='best')
        #plt.show()



    def test_SimpleSmoothLJ(self):
        eps = 1.7294663266397667
        sig = 3.253732668164946
        pot = LJ93SimpleSmooth(eps, sig, 3*sig)

        # import matplotlib.pyplot as plt
        # plt.figure()
        # r = np.linspace(pot.r_min*.7, pot.r_c*1.1, 100)
        # p = pot.evaluate(r)[0]
        # plt.plot(r, p)
        # pois = [pot.r_c, pot.r_min]
        # plt.scatter(pois, pot.evaluate(pois)[0])
        # plt.ylim(bottom=1.1*p.min(), top=-.3*p.min())
        # plt.grid(True)
        # plt.legend(loc='best')
        # plt.show()

    def test_SimpleSmoothVDW(self):
        hamaker = 68.1e-21
        c_sr = 2.1e-78*1e-6
        r_c = 10e-10
        pot = VDW82SimpleSmooth(c_sr, hamaker, 10e-10)

        # import matplotlib.pyplot as plt
        # r = np.linspace(pot.r_min*.7, pot.r_c*1.1, 1000)
        # ps = pot.evaluate(r, pot=True, forces=True)
        #
        # for i, name in enumerate(('potential', 'force')):
        #     plt.figure()
        #     p = ps[i]
        #     plt.plot(r, p, label=name)
        #
        #     pois = [pot.r_c, pot.r_min]
        #     plt.scatter(pois, pot.evaluate(pois, pot=True, forces=True)[i])
        #     plt.ylim(bottom=1.1*p.min(), top=-.3*p.min())
        #     plt.grid(True)
        #     plt.legend(loc='best')
        # plt.show()

    def test_ExpPotential(self):
        r = np.linspace(-10, 10, 1001)
        pot = ExpPotential(1.0, 1.0)
        V, dV, ddV = pot.naive_pot(r)
        self.assertTrue((V<0.0).all())
        self.assertTrue((dV<0.0).all())
        self.assertTrue((ddV<0.0).all())
        dV_num = np.diff(V)/np.diff(r)
        ddV_num = np.diff(dV_num)/(r[1]-r[0])
        self.assertLess(abs((dV[:-1]+dV[1:])/2+dV_num).max(), 1e-4)
        self.assertLess(abs(ddV[1:-1]-ddV_num).max(), 1e-2)

    def test_rinfl(self):
        """
        Test if the inflection point calculated analyticaly is really a 
        signum change of the second dericative
        """

        eps = 1.7294663266397667
        sig = 3.253732668164946

        c_sr = 2.1e-78
        hamaker = 68.1e-21

        all_ok = True
        msg = []
        for pot in [
            LJ93(eps, sig),
            LJ93SimpleSmooth(eps, sig, 3*sig),
            LJ93smooth(eps, sig),
            LJ93smoothMin(eps, sig),
            LJ93smooth(eps,  sig, r_t="inflection"),
            LJ93smoothMin(eps, sig, r_t_ls="inflection"),
            LJ93smooth(eps,  sig, r_t=LJ93(eps, sig).r_infl*1.05),
            LJ93smoothMin(eps,  sig, r_t_ls=LJ93(eps, sig).r_infl*1.05),
            VDW82(c_sr, hamaker),
            VDW82smooth(c_sr,  hamaker),
            VDW82smoothMin(c_sr,  hamaker),
            VDW82smooth(c_sr,  hamaker, r_t="inflection"),
            VDW82smoothMin(c_sr,  hamaker, r_t_ls="inflection"),
            VDW82smooth(c_sr,  hamaker, r_t=VDW82(c_sr, hamaker).r_infl * 1.05),
            DW82smoothMin(c_sr,  hamaker, r_t_ls=VDW82(c_sr, hamaker).r_infl*1.05)
                    ]:
            
            ok = (pot.evaluate(pot.r_infl * (1-1e-5), True, True, True)[2]
                  * pot.evaluate(pot.r_infl * (1+1e-5), True, True, True)[2] < 0)
            all_ok &= ok
            msg.append("{} \n {} ".format(pot, ok))

        self.assertTrue(all_ok, "\n"+"\n\n".join(msg))


