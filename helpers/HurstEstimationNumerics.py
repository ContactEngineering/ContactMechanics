#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   HurstEstimationNumerics.py

@author Till Junge <till.junge@epfl.ch>

@date   08 Jul 2015

@brief  Tests to understand the difficulties in extracting hurst from noisy data

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
import matplotlib.pyplot as plt

import PyPyContact.Tools as Tools
import PyPyContact.Surface as Surf


def plot_naive(surface, lam_max):
    fig = plt.figure()
    ax=fig.add_subplot(111)
    ax.set_yscale('log')
    ax.set_xscale('log')

    surf = Tools.CharacterisePeriodicSurface(surface)
    q = surf.q
    C = surf.C
    H, alpha = surf.estimate_hurst_naive(lambda_max=lam_max, full_output=True)
    print("H = {}, alpha = {}".format(H, alpha))
    ax.loglog(q, C, alpha=.1)
    mean, err, q_g = surf.grouped_stats(100)
    mask = np.isfinite(mean)
    mean = mean[mask]
    err = err[:, mask]
    q_g = q_g[mask]

    ax.errorbar(q_g, mean, yerr=err)
    ax.set_title("Naive: H={:.2f}, h_rms={:.2e}".format(H, np.sqrt((surface.profile()**2).mean())))
    a, b = np.polyfit(np.log(q), np.log(C), 1)
    ax.plot(q, q**(-2-2*H)*alpha, label="{}, H={:.2f}".format('fit', H))
    ax.legend(loc='best')

def plot_grad_C0(surface, H_in, lam_max):
    surf = Tools.CharacterisePeriodicSurface(surface)
    q_min = 2*np.pi/lam_max
    sl = surf.q > q_min
    q = surf.q[sl]
    C = surf.C[sl]
    dim = 2

    def C0_of_H(H):
        return ((q**(-3-2*H) * C).sum() /
            (q**(-5-4*H)).sum())

    def objective(H, C0):
        return ((C - C0*q**(-2*H-2))**2 /
                q**(dim-1)).sum()

    C0 = C0_of_H(H_in)
    O0 = objective(H_in, C0)
    c_s = np.linspace(0, 2*C0, 51)
    o_s = np.zeros_like(c_s)

    for i, c in enumerate(c_s):
        o_s[i] = objective(H_in, c)

    fig = plt.figure()
    ax=fig.add_subplot(111)
    fig.suptitle('grad(C0)')
    ax.plot(c_s, o_s, marker= '+')
    ax.scatter(C0, O0, marker='x', label = 'root', c='r')
    ax.grid(True)
    print("C0 = {}, obj0 = {}".format(C0, O0))
    return C0

def plot_grad_H(surface, lam_max):
    surf = Tools.CharacterisePeriodicSurface(surface)
    q_min = 2*np.pi/lam_max
    sl = surf.q > q_min
    q = surf.q[sl]
    C = surf.C[sl]
    dim = 2

    def C0_of_H(H):
        return ((q**(-2-(dim-1)-2*H) * C).sum() /
            (q**(-4-(dim-1)-4*H)).sum())

    def grad_h(H, C0):
        #return (4*C0*np.log(q)*(C-C0*q**(-2-2*H))/q).sum()
        #return (4*q**(-4*H)*(C*q**(2*H+2) - C0)*np.log(q)*C0/q**(4+(dim-1))).sum()
        return (4*C0*np.log(q)*q**(-1-2*H-dim)*(C - C0*q**(-2-2*H))).sum()

    def objective(H, C0):
        return ((C - C0*q**(-2*H-2))**2 /
                q**(dim-1)).sum()

    def full_obj(H):
        C0 = C0_of_H(H)
        return ((C - C0*q**(-2*H-2))**2 /
                q**(dim-1)).sum()

    h_s = np.linspace(.0, 2., 51)
    o_s = np.zeros_like(h_s)
    g_s = np.zeros_like(h_s)

    for i, h in enumerate(h_s):
        c = C0_of_H(h)
        o_s[i] = objective(h, c)
        g_s[i] = grad_h(h, c)

    H_opt, obj_opt, err, nfeq = scipy.optimize.fminbound(full_obj, 0, 2, full_output=True)
    if err != 0:
        raise Exception()

    fig = plt.figure()
    ax=fig.add_subplot(211)
    ax.set_xlim(h_s[0], h_s[-1])
    fig.suptitle('grad(H)')
    ax.plot(h_s, o_s, marker= '+')
    ax.grid(True)
    ax.scatter(H_opt, obj_opt, marker='x', label = 'root', c='r')
    ax=fig.add_subplot(212)
    ax.set_xlim(h_s[0], h_s[-1])

    ax.plot(h_s, g_s, marker= '+')
    grad_opt = grad_h(H_opt, C0_of_H(H_opt))
    ax.scatter(H_opt, grad_opt, marker='x', label = 'root', c='r')
    #res = scipy.optimize.fmin
    #print("H_out = {}, obj0 = {}".format(C0, O0))
    ax.grid(True)
    return H_opt, C0_of_H(H_opt)

def compare_to_PyPy(surface, lam_max, H_ref, C0_ref):
    fig = plt.figure()
    ax=fig.add_subplot(111)
    ax.set_yscale('log')
    ax.set_xscale('log')

    surf = Tools.CharacterisePeriodicSurface(surface)
    q_min = 2*np.pi/lam_max
    sl = surf.q > q_min
    q = surf.q
    C = surf.C
    H, alpha, res = surf.estimate_hurst_alt(lambda_max=lam_max, full_output=True)
    print("H = {}, alpha = {}".format(H, alpha))
    ax.loglog(q, C, alpha=.1)
    mean, err, q_g = surf.grouped_stats(100)
    mask = np.isfinite(mean)
    mean = mean[mask]
    err = err[:, mask]
    q_g = q_g[mask]

    ax.errorbar(q_g, mean, yerr=err)
    ax.set_title("New: H_pypy={:.2f}, H_ref = {:.2f}, h_rms={:.2e}".format(H, H_ref, np.sqrt((surface.profile()**2).mean())))
    
    ax.plot(q[sl], q[sl]**(-2-2*H)*alpha, label="{}, H={:.4f}".format('fit', H), lw = 3)
    ax.plot(q[sl], q[sl]**(-2-2*H_ref)*C0_ref, label="{}, H={:.4f}".format('ref_fit', H_ref), lw = 3)
    ax.legend(loc='best')

def main():
    siz = 2000e-9
    lam_max = .2*siz

    size = (siz, siz)
    hurst = .75
    h_rms = 3.24e-8
    res = 128
    resolution = (res, res)

    seed = 2

    surface = Tools.RandomSurfaceGaussian(
        resolution, size, hurst, h_rms, lambda_max=lam_max, seed=seed).get_surface()

    plot_naive(surface, lam_max)

    plot_grad_C0(surface, hurst, lam_max)

    H, C0 = plot_grad_H(surface, lam_max)
    print("H_ref = {}, C0_ref = {}".format(H, C0))
    compare_to_PyPy(surface, lam_max, H, C0)


if __name__ == "__main__":
    main()
    plt.show()
