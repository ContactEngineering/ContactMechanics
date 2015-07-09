#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   SurfaceAnalysis.py

@author Till Junge <till.junge@kit.edu>

@date   25 Jun 2015

@brief  Provides a tools for the analysis of surface power spectra

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
import numpy as np
import scipy
from . import compute_wavevectors, fftn

class CharacterisePeriodicSurface(object):
    """
    Simple inverse FFT analysis without window. Do not use for measured surfs
    """
    eval_at_init=True
    def __init__(self, surface):
        """
        Keyword Arguments:
        surface -- Instance of PyPyContact.Surface or subclass with specified
                   size
        """
        self.surface = surface
        if self.surface.size is None:
            raise Exception("Surface size has to be known (and specified)!")
        if self.surface.dim != 2:
            raise Exception("Only 2D surfaces, for the time being")
        if self.surface.size[0] != self.surface.size[1]:
            raise Exception("Only square surfaces, for the time being")
        if self.surface.resolution[0] != self.surface.resolution[1]:
            raise Exception("Only square surfaces, for the time being")

        self.window=1
        if self.eval_at_init:
            self.C, self.q = self.eval()
            self.size = self.C.size

    def eval(self):
        res, size = self.surface.resolution, self.surface.size
        # equivalent lattice constant**2
        area = np.prod(size)
        h_a = fftn(self.surface.profile()*self.window, area)
        C_q = 1/area*(np.conj(h_a)*h_a).real

        q_vecs = compute_wavevectors(res, size, self.surface.dim)
        q_norm = np.sqrt((q_vecs[0]**2).reshape((-1, 1))+q_vecs[1]**2)
        order = np.argsort(q_norm, axis=None)
        # The first entry (for |q| = 0) is rejected, since it's 0 by construction
        return C_q.flatten()[order][1:], q_norm.flatten()[order][1:]

    def get_q_from_lambda(self, lambda_min, lambda_max):
        if lambda_min == 0:
            q_max = float('inf')
        else:
            q_max = 2*np.pi/lambda_min
        q_min = 2*np.pi/lambda_max
        return q_min, q_max


    def estimate_hurst_naive(self, lambda_min = 0, lambda_max = float('inf'),
                             full_output=False):
        """
        Naive way of estimating hurst exponent. biased towards short wave
        lengths, here only for legacy purposes
        """
        q_min, q_max = self.get_q_from_lambda(lambda_min, lambda_max)
        sl = np.logical_and(self.q<q_max, self.q>q_min)
        exponent, offset = np.polyfit(np.log(self.q[sl]),
                                      np.log(self.C[sl]), 1)
        prefactor = np.exp(offset)
        Hurst= -(exponent+2)/2
        if full_output:
            return Hurst, prefactor
        else:
            return Hurst


    def estimate_hurst(self, H_guess=1., C0_guess=None,
                       lambda_min=0, lambda_max = float('inf'),
                       full_output=False, tol = 1e-9, factor=1):
        """ When estimating Hurst for  more-than-one-dimensional surfs, we need to scale. E.g, 2d
        C(q) = C_0*q^(-2-2H)
        H, C_0 = argmin sum_i[|C(q_i)-self.C|^2/q_i]
        """
        q_min, q_max = self.get_q_from_lambda(lambda_min, lambda_max)
        sl = np.logical_and(self.q<q_max, self.q>q_min)
        # normalising_factor
        k_norm = factor/self.C[sl].sum()
        def objective(X):
            H, C0 = X
            return k_norm * ((self.C[sl] - C0*self.q[sl]**(-2-H*2))**2 /
                    self.q[sl]**(self.surface.dim-1)).sum()


        def jacobian(X):
            H, C0 = X
            sim = self.q[sl]**(-2-H*2)
            bracket = (self.C[sl] - C0*sim)
            denom = self.q[sl]**(self.surface.dim-1)
            return np.array([(4*C0 * sim * bracket * np.log(self.q[sl]) /
                              denom).sum(),
                             (-2*sim*bracket/denom).sum()])*k_norm
        H0 = H_guess
        if C0_guess is None:
            C0 = (self.C[sl]/self.q[sl]**(-2-H0*2)).mean()
        else:
            C0 = C0_guess
        res = scipy.optimize.minimize(objective, [H0, C0], tol=tol)#, jac=jacobian)
        if not res.success:
            raise Exception(
                ("Estimation of Hurst exponent did not succeed. Optimisation "
                 "result is :\n{}").format(res))
        Hurst, prefactor = res.x
        if full_output:
            return Hurst, prefactor, res
        else:
            return Hurst

    def estimate_hurst_alt(self, H_bracket=(0., 2.),
                       lambda_min=0, lambda_max = float('inf'),
                       full_output=False, tol = 1e-9):
        """ When estimating Hurst for  more-than-one-dimensional surfs, we need
        to scale. E.g, 2d
        C(q) = C_0*q^(-2-2H)
        H, C_0 = argmin sum_i[|C(q_i)-self.C|^2/q_i]
        """
        q_min, q_max = self.get_q_from_lambda(lambda_min, lambda_max)
        sl = np.logical_and(self.q<q_max, self.q>q_min)

        q = self.q[sl]
        C = self.C[sl]
        factor = 1.#/(C**2/q).sum()

        # The unique root of the gradient of the objective in C0 can be
        # explicitly expressed
        def C0_of_H(H):
            return ((q**(-3-2*H) * C).sum() /
                    (q**(-5-4*H)).sum())

        #this is the gradient of the objective in H
        def grad_in_H(H):
            C0 = C0_of_H(H)
            return ((4*q**(-4*H) *
                    (C*q**(2*H+2) -
                     C0)*np.log(q)*C0/q**5).sum())*factor


        def obj(H):
            C0 = C0_of_H(H)
            return (((C-C0*q**(-2-2*H))**2/q).sum())*factor

        h_s = np.linspace(H_bracket[0], H_bracket[1], 51)
        o_s = np.zeros_like(h_s)
        g_s = np.zeros_like(h_s)
        for i, h in enumerate(h_s):
            o_s[i] = obj(h)
            g_s[i] = grad_in_H(h)


        res = scipy.optimize.fminbound(
            obj, 0, 2, xtol=tol, full_output=True)#y, jac=grad_in_H)
        H_opt, obj_opt, err, nfeq = res
        if not err == 0:
            raise Exception(
                ("Estimation of Hurst exponent did not succeed. Optimisation "
                 "result is :\n{}").format(res))
        Hurst = H_opt

        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(211)
        ax.grid(True)
        ax.set_xlim(H_bracket)
        ax.plot(h_s, o_s)
        ax.scatter(Hurst, obj(Hurst), marker = 'x')
        ax = fig.add_subplot(212)
        ax.plot(h_s, g_s)
        ax.grid(True)
        ax.set_xlim(H_bracket)
        ax.scatter(Hurst, grad_in_H(Hurst), marker = 'x')

        if full_output:
            prefactor = C0_of_H(Hurst)
            return Hurst, prefactor, res
        else:
            return Hurst



    def compute_h_rms(self):
        return self.surface.compute_h_rms()

    def compute_h_rms_fromReciprocSpace(self):
        return self.surface.compute_h_rms_fromReciprocSpace()

    def grouped_stats(self, nb_groups, percentiles=(5, 95)):
        """
        Make nb_groups groups of the ordered C-q dataset and compute their means
        standard errors for plots with errorbars
        Keyword Arguments:
        nb_groups --
        """
        boundaries = np.logspace(np.log10(self.q[0]),
                                 np.log10(self.q[-1]), nb_groups+1)
        C_g = np.zeros(nb_groups)
        C_g_std = np.zeros((len(percentiles), nb_groups))
        q_g = np.zeros(nb_groups)

        for i in range(nb_groups):
            bottom = boundaries[i]
            top = boundaries[i+1]
            sl = np.logical_and(bottom <= self.q, self.q <= top)
            if sl.sum():
                C_sample = self.C[sl]
                q_sample = self.q[sl]

                C_g[i] = C_sample.mean()
                C_g_std[:, i] = abs(np.percentile(C_sample, percentiles)-C_g[i])
                q_g[i] = q_sample.mean()
            else:
                C_g[i] = float('nan')
                C_g_std[:, i] = float('nan')
                q_g[i] = float('nan')

            bottom = top

        return C_g, C_g_std, q_g

class CharacteriseSurface(CharacterisePeriodicSurface):
    """
    inverse FFT analysis with window. For analysis of measured surfaces
    """
    eval_at_init=False
    def __init__(self, surface, window_type='hanning', window_params=dict()):
        """
        Keyword Arguments:
        surface       -- Instance of PyPyContact.Surface or subclass with
                         specified size
        window_type   -- (default 'hanning') numpy windowing function name
        window_params -- (default dict())
        """
        super().__init__(surface)
        self.window = self.get_window(window_type, window_params)
        self.C, self.q = self.eval()
        self.size = self.C.size

    def get_window(self, window_type, window_params):
        if window_type == 'hanning':
            window = np.hanning(self.surface.resolution[0])
        elif window_type == 'kaiser':
            window = np.kaiser(self.surface.resolution[0], **window_params)
        else:
            raise Exception("Window type '{}' not known.".format(window_type))
        window = window.reshape((-1, 1))*window
        return window
