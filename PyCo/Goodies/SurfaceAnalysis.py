#
# Copyright 2016, 2019 Lars Pastewka
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
Provides a tools for the analysis of surface power spectra
"""

from __future__ import absolute_import, division, print_function

import numpy as np
import scipy

from ..Tools.common import compute_wavevectors, fftn, get_q_from_lambda
from ..Topography import Topography


class CharacterisePeriodicSurface(object):
    """
    Simple inverse FFT analysis without window. Do not use for measured surfs
    """
    eval_at_init = True

    def __init__(self, surface, one_dimensional=False):
        """
        Keyword Arguments:
        surface -- Instance of PyCo.Topography or subclass with specified
                   size
        one_dimensional -- (default False). if True, evaluation of 1D (line-
                           scan) power spectrum is emulated
        """
        # pylint: disable=invalid-name
        self.surface = surface
        if self.surface.size is None:
            raise Exception("Topography size has to be known (and specified)!")
        if self.surface.dim != 2:
            raise Exception("Only 2D surfaces, for the time being")
        if self.surface.size[0] != self.surface.size[1]:
            raise Exception("Only square surfaces, for the time being")
        if self.surface.resolution[0] != self.surface.resolution[1]:
            raise Exception("Only square surfaces, for the time being")

        self.window = 1
        if self.eval_at_init:
            if one_dimensional:
                self.C, self.q = self.eval_1D()
            else:
                self.C, self.q = self.eval()
            self.size = self.C.size

    def eval(self):
        """
        Generates the phases and amplitudes, readies the metasurface to
        generate Topography objects
        """
        res, size = self.surface.resolution, self.surface.size
        # equivalent lattice constant**2
        area = np.prod(size)
        h_a = fftn(self.surface.heights() * self.window, area)
        C_q = 1/area*(np.conj(h_a)*h_a).real

        q_vecs = compute_wavevectors(res, size, self.surface.dim)
        q_norm = np.sqrt((q_vecs[0]**2).reshape((-1, 1))+q_vecs[1]**2)
        order = np.argsort(q_norm, axis=None)
        # The first entry (for |q| = 0) is rejected, since it's 0 by construct
        return C_q.flatten()[order][1:], q_norm.flatten()[order][1:]

    def eval_1D(self):  # pylint: disable=invalid-name
        """
        Generates the phases and amplitudes, readies the metasurface to
        generate Topography objects
        """
        res, size = self.surface.resolution, self.surface.size
        # equivalent lattice constant**2

        tmp = np.fft.fft(self.surface.heights() * self.window, axis=0)
        D_q_x = np.conj(tmp)*tmp
        D_q = np.mean(D_q_x, axis=1).real

        q_x = abs(compute_wavevectors(res, size, self.surface.dim)[0])
        print("q_x = {}".format(q_x))
        order = np.argsort(q_x, axis=None)
        print("order = {}".format(order))
        # The first entry (for |q| = 0) is rejected, since it's 0 by construct
        return D_q.flatten()[order][1:], q_x.flatten()[order][1:]

    def estimate_hurst(self, lambda_min=0, lambda_max=float('inf'),
                       full_output=False):
        """
        Naive way of estimating hurst exponent. biased towards short wave
        lengths, here only for legacy purposes
        """
        q_min, q_max = get_q_from_lambda(lambda_min, lambda_max)
        q_max = min(q_max, self.q_shannon)
        sl = np.logical_and(self.q < q_max, self.q > q_min)
        weights = np.sqrt(1/self.q[sl]/((1/self.q[sl]).sum()))
        # Note the weird definition of the weights here. this is due to
        # numpy's polyfit interface. Since I suspect that numpy will change
        # this, i avoid using polyfit
        # exponent, offset = np.polyfit(np.log(self.q[sl]),
        #                               np.log(self.C[sl]),
        #                               1,
        #                               w=np.sqrt(1/self.q[sl]))

        # and do it 'by hand'
        # pylint: disable=invalid-name
        A = np.vstack((np.log(self.q[sl])*weights, weights)).T
        exponent, offset = np.linalg.lstsq(A, np.log(self.C[sl])*weights, rcond=None)[0]
        C0 = np.exp(offset)
        Hurst = -(exponent+2)/2
        if full_output:
            return Hurst, C0
        else:
            return Hurst

    def estimate_hurst_bad(self, H_guess=1., C0_guess=None,
                           lambda_min=0, lambda_max=float('inf'),
                           full_output=False, tol=1e-9, factor=1):
        """ When estimating Hurst for  more-than-one-dimensional surfs, we need
        to scale. E.g, 2d
        C(q) = C_0*q^(-2-2H)
        H, C_0 = argmin sum_i[|C(q_i)-self.C|^2/q_i]
        """
        # pylint: disable=invalid-name
        q_min, q_max = get_q_from_lambda(lambda_min, lambda_max)
        q_max = max(q_max, self.q_shannon)
        sl = np.logical_and(self.q < q_max, self.q > q_min)
        # normalising_factor
        k_norm = factor/self.C[sl].sum()

        def objective(X):
            " fun to minimize"
            H, C0 = X
            return k_norm * ((self.C[sl] - C0*self.q[sl]**(-2-H*2))**2 /
                             self.q[sl]**(self.surface.dim-1)).sum()

        # def jacobian(X):
        #     " jac of objective"
        #     H, C0 = X
        #     sim = self.q[sl]**(-2-H*2)
        #     bracket = (self.C[sl] - C0*sim)
        #     denom = self.q[sl]**(self.surface.dim-1)
        #     return np.array([(4*C0 * sim * bracket * np.log(self.q[sl]) /
        #                       denom).sum(),
        #                      (-2*sim*bracket/denom).sum()])*k_norm
        H0 = H_guess
        if C0_guess is None:
            C0 = (self.C[sl]/self.q[sl]**(-2-H0*2)).mean()
        else:
            C0 = C0_guess
        res = scipy.optimize.minimize(objective, [H0, C0], tol=tol)
        #                             , jac=jacobian)
        if not res.success:
            raise Exception(
                ("Estimation of Hurst exponent did not succeed. Optimisation "
                 "result is :\n{}").format(res))
        Hurst, prefactor = res.x
        if full_output:
            return Hurst, prefactor, res
        else:
            return Hurst

    def estimate_hurst_from_mean(self, lambda_min=0, lambda_max=float('inf'),
                                 full_output=False):
        """
        Naive way of estimating hurst exponent. biased towards short wave
        lengths, here only for legacy purposes
        """
        # pylint: disable=invalid-name
        lambda_min = max(lambda_min, self.lambda_shannon)

        C_m, dummy, q_m = self.grouped_stats(100)
        q_min, q_max = get_q_from_lambda(lambda_min, lambda_max)
        sl = np.logical_and(q_m < q_max, q_m > q_min)
        A = np.ones((sl.sum(), 2))
        A[:, 0] = np.log(q_m[sl])
        exponent, offset = np.linalg.lstsq(A, np.log(C_m[sl]))[0]
        C0 = np.exp(offset)
        Hurst = -(exponent+2)/2
        if full_output:
            return Hurst, C0
        else:
            return Hurst

    def estimate_hurst_alt(self, H_bracket=(0., 2.),
                           lambda_min=0, lambda_max=float('inf'),
                           full_output=False, tol=1e-9):
        import matplotlib.pyplot as plt

        """ When estimating Hurst for  more-than-one-dimensional surfs, we need
        to scale. E.g, 2d
        C(q) = C_0*q^(-2-2H)
        H, C_0 = argmin sum_i[|C(q_i)-self.C|^2/q_i]
        """
        # pylint: disable=invalid-name
        q_min, q_max = get_q_from_lambda(lambda_min, lambda_max)
        sl = np.logical_and(self.q < q_max, self.q > q_min)

        q = self.q[sl]
        C = self.C[sl]
        factor = 1.  # /(C**2/q).sum()

        # The unique root of the gradient of the objective in C0 can be
        # explicitly expressed
        def C0_of_H(H):  # pylint: disable=missing-docstring
            return ((q**(-3-2*H) * C).sum() /
                    (q**(-5-4*H)).sum())

        # this is the gradient of the objective in H
        def grad_in_H(H):  # pylint: disable=missing-docstring
            C0 = C0_of_H(H)
            return ((4*q**(-4*H) *
                     (C*q**(2*H+2) -
                      C0)*np.log(q)*C0/q**5).sum())*factor

        def obj(H):  # pylint: disable=missing-docstring
            C0 = C0_of_H(H)
            return (((C-C0*q**(-2-2*H))**2/q).sum())*factor

        h_s = np.linspace(H_bracket[0], H_bracket[1], 51)
        o_s = np.zeros_like(h_s)
        g_s = np.zeros_like(h_s)
        for i, h in enumerate(h_s):
            o_s[i] = obj(h)
            g_s[i] = grad_in_H(h)

        res = scipy.optimize.fminbound(
            obj, 0, 2, xtol=tol, full_output=True)  # y, jac=grad_in_H)
        H_opt, dummy_obj_opt, err, dummy_nfeq = res
        if not err == 0:
            raise Exception(
                ("Estimation of Hurst exponent did not succeed. Optimisation "
                 "result is :\n{}").format(res))
        Hurst = H_opt

        fig = plt.figure()
        ax = fig.add_subplot(211)
        ax.grid(True)
        ax.set_xlim(H_bracket)
        ax.plot(h_s, o_s)
        ax.scatter(Hurst, obj(Hurst), marker='x')
        ax = fig.add_subplot(212)
        ax.plot(h_s, g_s)
        ax.grid(True)
        ax.set_xlim(H_bracket)
        ax.scatter(Hurst, grad_in_H(Hurst), marker='x')

        if full_output:
            prefactor = C0_of_H(Hurst)
            return Hurst, prefactor, res
        else:
            return Hurst

    def compute_rms_height(self):  # pylint: disable=missing-docstring
        return self.surface.rms_height()

    def compute_rms_slope(self):  # pylint: disable=missing-docstring
        return self.surface.rms_slope()

    def compute_rms_height_q_space(self):  # pylint: disable=missing-docstring
        tmp_surf = Topography(self.surface.heights * self.window,
                              size=self.surface.size)
        return tmp_surf.rms_height_q_space()

    def compute_rms_slope_q_space(self):  # pylint: disable=missing-docstring
        tmp_surf = Topography(self.surface.heights() * self.window,
                              size=self.surface.size)
        return tmp_surf.rms_slope_q_space()

    def grouped_stats(self, nb_groups, percentiles=(5, 95), filter_nan=True):
        """
        Make nb_groups groups of the ordered C-q dataset and compute their
        means standard errors for plots with errorbars
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
                C_sample = self.C[sl]  # pylint: disable=invalid-name
                q_sample = self.q[sl]

                C_g[i] = C_sample.mean()
                C_g_std[:, i] = abs(
                    np.percentile(C_sample, percentiles)-C_g[i])
                q_g[i] = q_sample.mean()
            else:
                C_g[i] = float('nan')
                C_g_std[:, i] = float('nan')
                q_g[i] = float('nan')

            bottom = top

        if filter_nan:
            sl = np.isfinite(q_g)
            return C_g[sl], C_g_std[:, sl], q_g[sl]
        return C_g, C_g_std, q_g

    @property
    def lambda_shannon(self):
        " wavelength of shannon limit"
        return 2*self.surface.size[0]/self.surface.resolution[0]

    @property
    def q_shannon(self):
        " angular freq of shannon limit"
        return 2*np.pi/self.lambda_shannon

    def simple_spectrum_plot(self, title=None, q_minmax=(0, float('inf')),
                             y_min=None, n_bins=100, ax=None, color='b',
                             errbar=True, x_max_at_shannon=False, fit_alpha=1.):
        " Convenience function to plot power spectra with informative labels"
        import matplotlib.pyplot as plt

        line_width = 3
        q_sh = self.q_shannon

        def corrector(q):  # pylint: disable=invalid-name,missing-docstring
            if q == 0:
                return float('inf')
            elif q == float('inf'):
                q = q_sh
            return 2*np.pi/q
        lam_max, lam_min = (corrector(q) for q in q_minmax)
        try:
            q_minmax = tuple((2*np.pi/l for l in (lam_max, lam_min)))
        except TypeError as err:
            raise TypeError(
                "{}: lam_max = {}, lam_min = {}, q_minmax = {}".format(
                    err, lam_max, lam_min, q_minmax))
        Hurst, C0 = self.estimate_hurst_from_mean(lambda_min=lam_min,
                                                  lambda_max=lam_max,
                                                  full_output=True)
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.grid(True)
        else:
            fig = None
        mean, err, q_g = self.grouped_stats(n_bins)
        fit_q = q_g[np.logical_and(q_g < q_minmax[1], q_g > q_minmax[0])]
        if errbar:
            ax.errorbar(q_g, mean, yerr=err, color=color)
        else:
            ax.plot(q_g, mean, color=color)
        if title is not None:
            ax.set_title(title)
        ax.loglog(fit_q, C0*fit_q**(-2-2*Hurst), color='r', lw=line_width,
                  label=(r"$H = {:.3f}$, $C_0 = {:.3e}$".format(Hurst, C0) +
                         r""),
                  alpha=fit_alpha)
        ax.set_xlabel(r"$\left|q\right|$ in [rad/m]")
        ax.set_ylabel(r"$C(\left|q\right|)$ in [$\mathrm{m}^4$]")
        ax.set_yscale('log')
        ax.set_xscale('log')
        if not x_max_at_shannon:
            ylims = ax.get_ylim()
            ax.plot((q_sh, q_sh), ylims, color='k',
                    label=r'$\lambda_\mathrm{Shannon}$')
        else:
            ax.set_xlim(right=q_sh)
        if y_min is not None:
            ax.set_ylim(bottom=y_min)
        if fig is not None:
            ax.legend(loc='best')
            fig.subplots_adjust(left=.19, bottom=.16, top=.85)
        return fig, ax


class CharacteriseSurface(CharacterisePeriodicSurface):
    """
    inverse FFT analysis with window. For analysis of measured surfaces
    """
    eval_at_init = False

    def __init__(self, surface, window_type='hanning', window_params=None):
        """
        Keyword Arguments:
        surface       -- Instance of PyCo.Topography or subclass with
                         specified size
        window_type   -- (default 'hanning') numpy windowing function name
        window_params -- (default dict())
        """
        super().__init__(surface)
        if window_params is None:
            window_params = dict()
        self.window = self.get_window(window_type, window_params)
        self.C, self.q = self.eval()
        self.size = self.C.size

    def get_window(self, window_type, window_params):
        " return an evaluated window as an np.array"
        if window_type == 'hanning':
            window = 2*np.hanning(self.surface.resolution[0])
        elif window_type == 'kaiser':
            window = np.kaiser(self.surface.resolution[0], **window_params)
        else:
            raise Exception("Window type '{}' not known.".format(window_type))
        window = window.reshape((-1, 1))*window
        return window