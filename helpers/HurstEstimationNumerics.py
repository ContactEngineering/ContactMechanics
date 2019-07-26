#
# Copyright 2019 Antoine Sanner
#           2018-2019 Lars Pastewka
#           2015-2016 Till Junge
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
Tests to understand the difficulties in extracting hurst from noisy data
"""
import numpy as np
import scipy
import matplotlib.pyplot as plt

import PyCo.Tools as Tools
import PyCo.Topography as Surf


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
    ax.set_title("Naive: H={:.2f}, h_rms={:.2e}".format(H, np.sqrt((surface.heights() ** 2).mean())))
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
        return ((q**(-3-2*H)).sum() /
            (q**(-5-4*H)/C).sum())

    def objective(H, C0):
        return ((1 - C0*q**(-2*H-2)/C)**2 /
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
    q = surf.q[sl]# np.array(surf.q[sl][0], surf.q[sl][-1])
    C = surf.C[sl]# np.array(surf.C[sl][0], surf.C[sl][-1])
    dim = 2

    def C0_of_H(H):
        return ((C**2/q**(-5-dim-4*H)).sum() /
            (C/q**(-3-dim-2*H)).sum())

    def grad_h(H, C0):
        return (4*C0/C*np.log(q)*q**(-1-2*H-dim)*(1 - C0*q**(-2-2*H)/C)).sum()

    def objective(H, C0):
        return ((c/q**(-2*H-2) - C0)**2 /
                q**(dim-1)).sum()

    def full_obj(H):
        C0 = C0_of_H(H)
        return ((1 - C0/C*q**(-2*H-2))**2 /
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
    ax.set_title("New: H_pypy={:.2f}, H_ref = {:.2f}, h_rms={:.2e}".format(H, H_ref, np.sqrt((surface.heights() ** 2).mean())))
    
    ax.plot(q[sl], q[sl]**(-2-2*H)*alpha, label="{}, H={:.4f}".format('fit', H), lw = 3)
    ax.plot(q[sl], q[sl]**(-2-2*H_ref)*C0_ref, label="{}, H={:.4f}".format('ref_fit', H_ref), lw = 3)
    ax.legend(loc='best')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.loglog(q[sl], C[sl]/(q[sl]**(-2-2*H_ref)*C0_ref), alpha=.1)
    ax.errorbar(q_g, mean/(q_g**(-2-2*H_ref)*C0_ref), yerr=err/(q_g**(-2-2*H_ref)*C0_ref))

def main():
    siz = 2000e-9
    lam_max = .2*siz

    size = (siz, siz)
    hurst = .75
    h_rms = 3.24e-8
    res = 128
    nb_grid_pts = (res, res)

    seed = 2

    surface = Tools.RandomSurfaceGaussian(
        nb_grid_pts, size, hurst, h_rms, lambda_max=lam_max, seed=seed).get_surface()

    plot_naive(surface, lam_max)

    plot_grad_C0(surface, hurst, lam_max)

    H, C0 = plot_grad_H(surface, lam_max)
    print("H_ref = {}, C0_ref = {}".format(H, C0))
    compare_to_PyPy(surface, lam_max, H, C0)


if __name__ == "__main__":
    main()
    plt.show()
