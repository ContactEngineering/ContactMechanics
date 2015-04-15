#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   Potential.py

@author Till Junge <till.junge@kit.edu>

@date   21 Jan 2015

@brief  Generic potential class, all potentials inherit it

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

from .Interactions import SoftWall
import math
import numpy as np
import scipy.optimize
import abc


class Potential(SoftWall, metaclass=abc.ABCMeta):
    """ Describes the minimum interface to interaction potentials for
        PyPyContact. These pontentials are purely 1D, which allows for a few
        simplifications. For instance, the potential energy and forces can be
        computed at any point in the problem from just the one-dimensional gap
        (h(x,y)-z(x,y)) at that point
    """
    name = "generic_potential"

    class PotentialError(Exception):
        "umbrella exception for potential-related issues"
        pass

    class SliceableNone(object):
        """small helper class to remedy numpy's lack of views on
        index-sliced array views. This construction avoid the computation
        of all interactions as with np.where, and copies"""
        # pylint: disable=too-few-public-methods
        __slots__ = ()

        def __setitem__(self, index, val):
            pass

        def __getitem__(self, index):
            pass

    @abc.abstractmethod
    def __init__(self, r_cut):
        super().__init__()
        self.r_c = r_cut
        if r_cut is not None:
            self.has_cutoff = not math.isinf(self.r_c)
        else:
            self.has_cutoff = False
        if self.has_cutoff:
            self.offset = self.naive_pot(self.r_c)[0]
        else:
            self.offset = 0

        self.curb = None

    @abc.abstractmethod
    def __repr__(self):
        return ("Potential '{0.name}', cut-off radius r_cut = " +
                "{0.r_c}").format(self)

    def compute(self, gap, pot=True, forces=False, curb=False, area_scale=1.):
        # pylint: disable=arguments-differ
        energy, self.force, self.curb = self.evaluate(
            gap, pot, forces, curb, area_scale)
        self.energy = energy.sum() if pot else None

    @abc.abstractmethod
    def naive_pot(self, r, pot=True, forces=False, curb=False):
        """ Evaluates the potential and its derivatives without cutoffs or
            offsets.
            Keyword Arguments:
            r      -- array of distances
            pot    -- (default True) if true, returns potential energy
            forces -- (default False) if true, returns forces
            curb   -- (default False) if true, returns second derivative

        """
        raise NotImplementedError()

    def evaluate(self, r, pot=True, forces=False, curb=False, area_scale=1.):
        """Evaluates the potential and its derivatives
        Keyword Arguments:
        r          -- array of distances
        pot        -- (default True) if true, returns potential energy
        forces     -- (default False) if true, returns forces
        curb       -- (default False) if true, returns second derivative
        area_scale -- (default 1.) scale by this. (Interaction quantities are
                      supposed to be expressed per unit area, so systems need
                      to be able to scale their response for their resolution))
        """
        # pylint: disable=bad-whitespace
        # pylint: disable=arguments-differ
        r = np.array(r)
        if r.shape == ():
            r.shape = (1, )
        inside_slice = r < self.r_c
        V = np.zeros_like(r) if pot else self.SliceableNone()
        dV = np.zeros_like(r) if forces else self.SliceableNone()
        ddV = np.zeros_like(r) if curb else self.SliceableNone()

        V[inside_slice], dV[inside_slice], ddV[inside_slice] = self.naive_pot(
            r[inside_slice], pot, forces, curb)
        if V[inside_slice] is not None:
            V[inside_slice] -= self.offset
        return (area_scale*V if pot else None,
                area_scale*dV if forces else None,
                area_scale*ddV if curb else None)

    @abc.abstractproperty
    def r_min(self):
        """
        convenience function returning the location of the enery minimum
        """
        raise NotImplementedError()

    @property
    def naive_min(self):
        """ convenience function returning the energy minimum of the bare
           potential

        """
        return self.naive_pot(self.r_min)[0]


class SmoothPotential(Potential):
    """
    implements the splining of the potential tail using a fourth order
    polynomial than reaches zero within a finite cutoff radius

    decoupled potential:
    V_lγ (r) = V_l(r) - V_l(r_t) + γ

    spline: (Δr = r-r_t)
                                2        3        4
                           C₂⋅Δr    C₃⋅Δr    C₄⋅Δr
    V_s(Δr) = C₀ - C₁⋅Δr - ────── - ────── - ──────
                             2        3        4

    The formulation allows to choose the contact stiffness (the original
    repulsive zone) independently from the work of adhesion (the energy
    well). By default, the work of adhesion equals -min(V(r)) and the
    transition point r_t between the naive potential and the  spline is at the
    minimumm, however they can be chosen freely.

    The spline is chosen to guarantee continuity of the second derivative
    of the potential, leading to the following conditions:
    (1): V_s' (Δr_t)    =  V_lγ' (r_t)
    (2): V_s''(Δr_t)    =  V_lγ''(r_t)
    (3): V_s  (Δr_c) =  0, where Δr_c = r_c-r_o
    (4): V_s' (Δr_c) =  0
    (5): V_s''(Δr_c) =  0
    (6): V_s  (Δr_m) = -γ, where Δr_m = r_min-r_o
    The unknowns are C_i (i in {0,..4}) and r_c
    r_o is the origin for the spline, which can be chosen freely. The original
    choice was r_o = r_t, but it turned out to be a bad choice, leading to a
    system of nonlinear equation in which only two of the six unknowns can be
    eliminated.

    With r_o = r_c, all coefficients C_i can be eliminated and a good initial
    guess for the remaining scalar nonlinear equation can be computed. see
    eval_poly_and_cutoff
    """
    def __init__(self, gamma=None, r_t=None):
        """
        Keyword Arguments:
        gamma   -- (default -V_min) Work of adhesion, defaults to the norm of
                   minimum of the potential. Note the sign. γ is assumed to be
                   non-negative value
        r_t     -- (default r_min) transition point, defaults to r_min (argmin
                   of pontential)
        """
        # pylint: disable=super-init-not-called
        # not calling the superclass's __init__ because this is used in diamond
        # inheritance and I do not want to have to worry about python's method
        # resolution order
        self.gamma = gamma if gamma is not None else -self.naive_min
        # Warning: this assumes that the minimum of the potential is a negative
        # value. This will fail curiously if you use this class to implement a
        # potential with a positive minimum
        if self.gamma < 0:
            raise self.PotentialError(
                ("γ should be a positive value."
                 "you specified and γ = {}.").format(
                     self.gamma))

        self.r_t = r_t if r_t is not None else self.r_min
        # coefficients of the spline
        self.coeffs = np.zeros(5)
        self.eval_poly_and_cutoff()

    @abc.abstractmethod
    def __repr__(self):
        raise NotImplementedError

    def evaluate(self, r, pot=True, forces=False, curb=False, area_scale=1.):
        """Evaluates the potential and its derivatives
        Keyword Arguments:
        r          -- array of distances
        pot        -- (default True) if true, returns potential energy
        forces     -- (default False) if true, returns forces
        curb       -- (default False) if true, returns second derivative
        area_scale -- (default 1.) scale by this. (Interaction quantities are
                      supposed to be expressed per unit area, so systems need
                      to be able to scale their response for their resolution))
        """
        # pylint: disable=bad-whitespace
        # pylint: disable=invalid-name
        r = np.array(r)
        nb_dim = len(r.shape)
        if nb_dim == 0:
            r.shape = (1,)
        V = np.zeros_like(r) if pot else self.SliceableNone()
        dV = np.zeros_like(r) if forces else self.SliceableNone()
        ddV = np.zeros_like(r) if curb else self.SliceableNone()

        sl_inner = r < self.r_t
        sl_rest = np.logical_not(sl_inner)
        # little hack to work around numpy bug
        if np.array_equal(sl_inner, np.array([True])):
            V, dV, ddV = self.naive_pot(r, pot, forces, curb)
            V -= self.offset
            return V, dV, ddV
        else:
            V[sl_inner], dV[sl_inner], ddV[sl_inner] = self.naive_pot(
                r[sl_inner], pot, forces, curb)
        V[sl_inner] -= self.offset

        sl_outer = np.logical_and(r < self.r_c, sl_rest)
        # little hack to work around numpy bug
        if np.array_equal(sl_outer, np.array([True])):
            V, dV, ddV = self.spline_pot(r, pot, forces, curb)
        else:
            V[sl_outer], dV[sl_outer], ddV[sl_outer] = self.spline_pot(
                r[sl_outer], pot, forces, curb)

        return (area_scale*V if pot else None,
                area_scale*dV if forces else None,
                area_scale*ddV if curb else None)

    def spline_pot(self, r, pot=True, forces=False, curb=False):
        """ Evaluates the spline part and its derivatives of the potential.
        Keyword Arguments:
        r      -- array of distances
        pot    -- (default True) if true, returns potential energy
        forces -- (default False) if true, returns forces
        curb   -- (default False) if true, returns second derivative
        """
        # pylint: disable=invalid-name
        V = dV = ddV = None
        dr = r-self.r_c
        if pot:
            V = self.poly(dr)
        if forces:
            dV = -self.dpoly(dr)  # Forces are the negative gradient
        if curb:
            ddV = self.ddpoly(dr)
        return (V, dV, ddV)

    def eval_poly_and_cutoff(self, xtol=1e-10):
        """
        Computes the coefficients C_i of the spline and the cutoff radius r_c
        based on the work of adhesion γ and the slope V'(r_t) and curvature
        V"(r_t) at the transition point. The calculations leading to the
        formulation I use here are done symbolically in SplineHelper.py. Check
        there for details. Short version:


                                     2        3        4
                                C₂⋅Δr    C₃⋅Δr    C₄⋅Δr
        V_s(Δr) = C₀ - C₁⋅Δr - ────── - ────── - ──────
                                  2        3        4

                                      2        3
        V_s'(Δr) = -C₁ - C₂⋅Δr - C₃⋅Δr  - C₄⋅Δr


                                          2
        V_s"(Δr) = -C₂ - 2⋅C₃⋅Δr - 3⋅C₄⋅Δr

        By applying the following boundary conditions
        (1): V_s'(Δr_t)    =  V_lγ'(r_t)
        (2): V_s"(Δr_t)    =  V_lγ"(r_t)
        (3): V_s (Δr_c) =  0, where Δr_c = r_c-r_o
        (4): V_s'(Δr_c) =  0
        (5): V_s"(Δr_c) =  0
        (6): V_s (Δr_m) = -γ, where Δr_m = r_min-r_o
        The unknowns are C_i (i in {0,..4}) and r_c
        and choosing the origin of the spline r_o at the cut-off r_c, the
        coefficients C_i can be determined explicitly (see SplineHelper.py):

        C₀: 0, C₁: 0, C₂: 0,

            -3⋅V'(r_t) + V"(r_t)⋅Δrt      2⋅V'(r_t) - V"(r_t)⋅Δrt
        C₃: ────────────────────────, C₄: ───────────────────────
                            2                            3
                         Δrt                          Δrt
        The remaining unknown is Δr_t (our proxy for r_c) but that equation has
        no analytical solution (I think), except for r_t = r_m. The equation
        is:
                                               4
                  Δrt⋅(3⋅V'(r_t) - V"(r_t)⋅Δrt)
        f(Δrt) = ────────────────────────────── + γ = 0
                                              3
                  12⋅(2⋅V'(r_t) - V"(r_t)⋅Δrt)

        When r_t = r_m (the default); the solution is
                      _________
                     ╱   3γ
        Δr_t = -2⋅  ╱  ───────
                  ╲╱   V"(r_m)

        this can be used as initial guess for solving f(Δrt). The derivative
        f'(Δrt) is to bulky to pretty-print here, so look at SplineHelper.py if
        you wish to see it.

        In conclusion, with the inputs of r_t, γ, V'(r_t) and V"(r_t), we can
        conpute the objective function f(Δrt) and its derivative f'(Δrt), which
        should be roughly constant around the solution (inspect output of
        SplineHelper.py). If r_m = r_t, we can directly compute Δrt. Else, with
        the additional input of V"(r_m), we can compute a good initial guess to
        solve the problem numerically.


        Keyword Arguments:
        xtol -- (default 1e-10) tolerance for numerical solution. Is scaled
                by γ internally.
        """
        dummy, force_t, ddV_t = self.naive_pot(self.r_t, pot=False,
                                               forces=True, curb=True)
        dV_t = -force_t
        dummy, dummy, ddV_m = self.naive_pot(self.r_min, pot=False,
                                             forces=False, curb=True)

        def spline(Δrt):
            " from SplineHelper.py"
            C_i = [0, 0, 0, (-3*dV_t + ddV_t*Δrt)/Δrt**2,
                   (2*dV_t - ddV_t*Δrt)/Δrt**3]
            # in numpy polynomial form
            polycoeffs = [-C_i[4]/4,
                          -C_i[3]/3,
                          0, 0, 0]
            return np.poly1d(polycoeffs)

        def inner_obj_fun(Δrt, gam_star):
            " from SplineHelper.py"
            return (Δrt*(3*dV_t - ddV_t*Δrt)**4/(12*(2*dV_t - ddV_t*Δrt)**3) +
                    gam_star)

        def inner_obj_derivative(Δrt, dummy):
            " from SplineHelper.py"
            return ((3*dV_t - ddV_t*Δrt)**3*(3*ddV_t*Δrt*(3*dV_t - ddV_t*Δrt) +
                                             (2*dV_t - ddV_t*Δrt) *
                                             (3*dV_t - 5*ddV_t*Δrt)) /
                    (12*(2*dV_t - ddV_t*Δrt)**4))

        def Δrm(Δrt):
            return Δrt*(3*dV_t - ddV_t*Δrt)/(2*dV_t - ddV_t*Δrt)

        # start with initial guess:
        Δrt0 = -2*np.sqrt(3*self.gamma/ddV_m)
        gam_star0 = self.gamma

        options = dict(xtol=self.gamma*xtol)
        Δrt = None

        def outer_obj_fun(gam):
            sol = scipy.optimize.root(
                inner_obj_fun, Δrt0, args=(gam,),
                jac=inner_obj_derivative, options=options)
            nonlocal Δrt
            if sol.success:
                Δrt = sol.x
                offset = self.naive_pot(self.r_t)[0] - \
                    np.array(spline(Δrt)(Δrt))
                error = self.naive_pot(self.r_min)[0] - offset + self.gamma

                return error
            else:
                err_str = ("Evaluation of spline for potential '{}' failed. "
                           "Please check whether the inputs make sense"
                           "").format(self)
                raise self.PotentialError(err_str)

        def outer_obj_derivative(gam):
            return np.array([1.])

        sol = scipy.optimize.root(
                outer_obj_fun, gam_star0,
                jac=outer_obj_derivative, options=options)

        if sol.success:
            self.r_c = self.r_t - Δrt
            self.poly = spline(Δrt)
            self.dpoly = np.polyder(self.poly)
            self.ddpoly = np.polyder(self.dpoly)
            self.offset = -(self.spline_pot(self.r_t)[0] -
                            self.naive_pot(self.r_t)[0])
        else:
            err_str = ("Evaluation of spline for potential '{}' failed. Please"
                       " check whether the inputs make sense").format(self)
            raise self.PotentialError(err_str)

    def eval_poly_and_cutoff_legacy(self, xtol=1e-10):
        """ Seems to be a bad method, do not use in general, will likely
            disappear.

            Computes the coefficients of the spline and the cutoff based on σ
            and ε. Since this is a non-linear system of equations, this
            requires some numerics.
            The equations derive from the continuity conditions in the class's
            docstring. A few simplifications can be made to conditions (1),
            (2), and (6):

            (1): C_1 = -V_l' (r_t)

            (2): C_2 = -V_l''(r_t)

            (6): C_0 = + γ + C_1*Δr_m + C_2/2*Δr_m^2
                       + C_3/3*Δr_m^3 + C_4/4*Δr_m^4
            Also, we have some info about the shape of the spline:
            since Δr_c is both an inflection point and an extremum and it has
            to be on the right of the minimum, the minimum has to be the
            leftmost extremum of the spline and therefore the global minimum.
            It follows:

            1) C_4 < 0
            2) we know sgn(C_1) and sgn(C_2)
            3) any reasonable choice of r_t leads to a Δr_c > 0

            from (5), we get the 2 possible solutions for Δr_c(C_2, C_3, C_4):
                   ⎡         ________________   ⎛        ________________⎞ ⎤
                   ⎢        ╱              2    ⎜       ╱              2 ⎟ ⎥
            Δr_c = ⎢-C₃ + ╲╱  -3⋅C₂⋅C₄ + C₃    -⎝C₃ + ╲╱  -3⋅C₂⋅C₄ + C₃  ⎠ ⎥
                   ⎢─────────────────────────, ────────────────────────────⎥
                   ⎣           3⋅C₄                        3⋅C₄            ⎦
            This however seems to lead to intractable polynomial equations
            (minutes of sympy without finding a solution, ran out of patience)

            Numerical approach: equations (1) and (2) are evauated immediately,
            Then the system of equations

                   ⎡                                   2         ⎤
                   ⎢          -C₂ - 2⋅C₃⋅Δrc - 3⋅C₄⋅Δrc          ⎥
                   ⎢                                             ⎥
                   ⎢                            2         3      ⎥
                   ⎢       -C₁ - C₂⋅Δrc - C₃⋅Δrc  - C₄⋅Δrc       ⎥
                   ⎢                                             ⎥
                   ⎢                       2         3         4 ⎥
            F(x) = ⎢                 C₂⋅Δrc    C₃⋅Δrc    C₄⋅Δrc  ⎥ = 0
                   ⎢   C₀ - C₁⋅Δrc - ─────── - ─────── - ─────── ⎥
                   ⎢                    2         3         4    ⎥
                   ⎢                                             ⎥
                   ⎢                    2         3         4    ⎥
                   ⎢              C₂⋅Δrm    C₃⋅Δrm    C₄⋅Δrm     ⎥
                   ⎢C₀ - C₁⋅Δrm - ─────── - ─────── - ─────── - γ⎥
                   ⎣                 2         3         4       ⎦
            is solved. Note that a positive work of adhesion (sticky surface)
            has a negative value for γ. The jacobian is:
                   ⎡                 2                                  ⎤
                   ⎢0  -2⋅Δrc  -3⋅Δrc           -2⋅C₃ - 6⋅C₄⋅Δrc        ⎥
                   ⎢                                                    ⎥
                   ⎢       2        3                               2   ⎥
                   ⎢0  -Δrc     -Δrc       -C₂ - 2⋅C₃⋅Δrc - 3⋅C₄⋅Δrc    ⎥
                   ⎢                                                    ⎥
                   ⎢       3       4                                    ⎥
            G(x) = ⎢   -Δrc    -Δrc                          2         3⎥
                   ⎢1  ──────  ──────   -C₁ - C₂⋅Δrc - C₃⋅Δrc  - C₄⋅Δrc ⎥
                   ⎢     3       4                                      ⎥
                   ⎢                                                    ⎥
                   ⎢       3       4                                    ⎥
                   ⎢   -Δrm    -Δrm                                     ⎥
                   ⎢1  ──────  ──────                  0                ⎥
                   ⎣     3       4                                      ⎦

            with x = [C₀  C₃  C₄  Δrc]

        Keyword Arguments:
        xtol -- tolerance for numerical solution. Is multiplied by ε
                internally.

        """
        # pylint: disable=bad-whitespace
        # pylint: disable=bad-continuation
        # pylint: disable=invalid-name
        # known coeffs
        dummy, dV, ddV = self.naive_pot(self.r_t, pot=False, forces=True,
                                        curb=True)
        C1 = self.coeffs[1] = -dV
        C2 = self.coeffs[2] = -ddV
        gam = -self.gamma
        r_t = self.r_t
        dr_m = self.r_min - r_t

        def obj_fun(x):
            "the root of this function defined the spline coefficients"
            C0, C3, C4, dr_c = x
            return np.array(
                [- C2 - 2*C3 * dr_c - 3*C4 * dr_c**2,
                 - C1 - C2 * dr_c - C3 * dr_c**2 - C4 * dr_c**3,
                 C0 - C1 * dr_c - C2/2 * dr_c**2 - C3/3*dr_c**3 - C4/4*dr_c**4,
                 C0 - C1 * dr_m - C2/2 * dr_m**2 - C3/3*dr_m**3 -
                 C4/4*dr_m**4 - gam])

        def jacobian(x):
            "evaluate the gradient of the objetive function"
            dummy, C3, C4, dr_c = x
            return np.array(
                [[0,    -2*dr_c, -3*dr_c**2,
                  -2*C3 - 6*C4*dr_c],
                 [0,   -dr_c**2,   -dr_c**3,
                  -C2 - 2*C3*dr_c - 3*C4*dr_c**2],
                 [1, -dr_c**3/3, -dr_c**4/4,
                  -C1 - C2*dr_c - C3*dr_c**2 - C4*dr_c**3],
                 [1, -dr_m**3/3, -dr_m**4/4,
                  0]])

        C3guess = -C2

        x0 = np.array([-gam, C3guess, 0, 2*self.r_min])
        options = dict(xtol=-gam*xtol)
        sol = scipy.optimize.root(obj_fun, x0, jac=jacobian, options=options)
        if sol.success:
            self.coeffs[0], self.coeffs[3], self.coeffs[4], self.r_c = sol.x
            self.r_c += self.r_t
            # !!WARNING!! poly is 'backwards': poly = [C4, C3, C2, C1, C0] and
            # all coeffs except C0 have the wrong sign, furthermore they are
            # divided by their order
            polycoeffs = [-self.coeffs[4]/4,
                          -self.coeffs[3]/3,
                          -self.coeffs[2]/2,
                          -self.coeffs[1],
                          +self.coeffs[0]]
            self.poly = np.poly1d(polycoeffs)
            self.dpoly = np.polyder(self.poly)
            self.ddpoly = np.polyder(self.dpoly)
            self.offset = -(self.spline_pot(self.r_t)[0] -
                            self.naive_pot(self.r_t)[0])
        else:
            err_str = ("Evaluation of spline for potential '{}' failed. Please"
                       " check whether the inputs make sense").format(self)
            raise self.PotentialError(err_str)


class MinimisationPotential(SmoothPotential):
    """
    Replaces the singular repulsive part of potentials by a linear part. This
    makes potentials maximally robust for the use with very bad initial
    parameters. Consider using this instead of loosing time guessing initial
    states for optimization.
    """
    def __init__(self, r_ti=None):
        """
        Keyword Arguments:
        r_ti    -- (default r_min/2) transition point between linear function
                   and lj, defaults to r_min
        """
        # pylint: disable=super-init-not-called
        # not calling the superclass's __init__ because this is used in diamond
        # inheritance and I do not want to have to worry about python's method
        # resolution order
        self.r_ti = r_ti if r_ti is not None else self.r_min/2
        self.lin_part = self.compute_linear_part()

    def compute_linear_part(self):
        " evaluates the two coefficients of the linear part of the potential"
        f_val, f_prime, dummy = super().evaluate(self.r_ti, True, True)
        return np.poly1d((float(-f_prime), f_val + f_prime*self.r_ti))

    @abc.abstractmethod
    def __repr__(self):
        raise NotImplementedError

    def evaluate(self, r, pot=True, forces=False, curb=False, area_scale=1.):
        """Evaluates the potential and its derivatives
        Keyword Arguments:
        r          -- array of distances
        pot        -- (default True) if true, returns potential energy
        forces     -- (default False) if true, returns forces
        curb       -- (default False) if true, returns second derivative
        area_scale -- (default 1.) scale by this. (Interaction quantities are
                      supposed to be expressed per unit area, so systems need
                      to be able to scale their response for their resolution))
        """
        # pylint: disable=bad-whitespace
        # pylint: disable=invalid-name
        r = np.array(r)
        nb_dim = len(r.shape)
        if nb_dim == 0:
            r.shape = (1,)
        V = np.zeros_like(r) if pot else self.SliceableNone()
        dV = np.zeros_like(r) if forces else self.SliceableNone()
        ddV = np.zeros_like(r) if curb else self.SliceableNone()

        sl_core = r < self.r_ti
        sl_rest = np.logical_not(sl_core)
        # little hack to work around numpy bug
        if np.array_equal(sl_core, np.array([True])):
            return self.lin_pot(r, pot, forces, curb)
        else:
            V[sl_core], dV[sl_core], ddV[sl_core] = self.lin_pot(
                r[sl_core], pot, forces, curb)

        sl_inner = np.logical_and(r < self.r_t, sl_rest)
        sl_rest *= np.logical_not(sl_inner)
        # little hack to work around numpy bug
        if np.array_equal(sl_inner, np.array([True])):
            V, dV, ddV = self.naive_pot(r, pot, forces, curb)
            V -= self.offset
            return V, dV, ddV
        else:
            V[sl_inner], dV[sl_inner], ddV[sl_inner] = self.naive_pot(
                r[sl_inner], pot, forces, curb)
        if pot:
            V[sl_inner] -= self.offset

        sl_outer = np.logical_and(r < self.r_c, sl_rest)
        # little hack to work around numpy bug
        if np.array_equal(sl_outer, np.array([True])):
            V, dV, ddV = self.spline_pot(r, pot, forces, curb)
        else:
            V[sl_outer], dV[sl_outer], ddV[sl_outer] = self.spline_pot(
                r[sl_outer], pot, forces, curb)

        return (area_scale*V if pot else None,
                area_scale*dV if forces else None,
                area_scale*ddV if curb else None)

    @abc.abstractmethod
    def naive_pot(self, r, pot=True, forces=False, curb=False):
        """ Evaluates the potential and its derivatives without cutoffs or
            offsets.
            Keyword Arguments:
            r      -- array of distances
            pot    -- (default True) if true, returns potential energy
            forces -- (default False) if true, returns forces
            curb   -- (default False) if true, returns second derivative

        """
        raise NotImplementedError()

    def lin_pot(self, r, pot=True, forces=False, curb=False):
        """ Evaluates the linear part and its derivatives of the potential.
        Keyword Arguments:
        r      -- array of distances
        pot    -- (default True) if true, returns potential energy
        forces -- (default False) if true, returns forces
        curb   -- (default False) if true, returns second derivative
        """
        V = None if pot is False else self.lin_part(r)
        dV = None if forces is False else -self.lin_part[1]
        ddV = None if curb is False else 0.
        return V, dV, ddV

    @abc.abstractproperty
    def r_min(self):
        """
        convenience function returning the location of the enery minimum
        """
        raise NotImplementedError()
