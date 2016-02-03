#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   l_bfgs_b_test.py

@author Till Junge <till.junge@kit.edu>

@date   13 Mar 2015

@brief  sample problem to get a grasp on how te precondition equaitons for
        scipy's l-bfgs-b solver

@section LICENCE

 Copyright (C) 2015 Till Junge

PyCo is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation, either version 3, or (at
your option) any later version.

PyCo is distributed in the hope that it will be useful, but
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
import scipy.optimize
import matplotlib.pyplot as plt
from PyCo.ContactMechanics import LJ93smoothMin as lj


pot = lj(epsilon=1., sigma = 5, gamma = 10)

def fun_gen_parabola(scale):
    poly = np.poly1d([1, 2, 3])
    dpoly = poly.deriv()
    def objective(x):
        v, dv = (poly(scale*x), dpoly(scale*x)*scale)
        return np.array(v), np.array(dv)
    return objective

def fun_gen_lj(scale):
    def objective(x):
        v, dv, dummy = pot.evaluate(scale*x, pot=True, forces=True, curb=False)
        return np.array(v), np.array(-dv)
    return objective

def fun_gen_substrate(scale):
    pot = lj(epsilon=1., sigma = 5, gamma = 10)
    x


def tester(fun_gen, x, x0):
    for method in ("CG", 'bfgs', 'l-bfgs-b'):
        scales = list()
        vals = list()
        for i in range(-20, 40):
            scale = 2**i
            scales.append(scale)
            fun = fun_gen(scale)
            x_use = x/scale
            if False:
                plt.figure()
                v, dv = fun(x_use)
                plt.plot(x, v)
                plt.plot(x, dv)
                eval_idx = (len(x)*3)//4
                plt.plot(x, dv[eval_idx]*(x_use-x_use[eval_idx])+v[eval_idx])
            try:
                res = scipy.optimize.minimize(fun, jac=True, x0=x0/scale, method=method)
                if isinstance(res.message, bytes):
                    res.message = res.message.decode()
                print(res.x*scale, res.fun, res.message, res.jac)
                vals.append((res.success, res.message))
            except Exception as err:
                print(err)
                vals.append((False, err))

        outcomes = set((mes for _, mes in vals))
        layers = {outcome:np.zeros(len(scales)) for outcome in outcomes}
        for i, (scale, (succ, outcome)) in enumerate(zip(scales, vals)):
            factor = 1 if succ else -1
            layers[outcome][i] = factor
        fig, ax1 = plt.subplots()
        ax1.set_yscale('log')
        scales = np.array(scales)
        print(layers.keys())
        colors = ('b', 'r', 'g', 'k','b', 'r', 'g', 'k','b', 'r', 'g', 'k',)
        plots = list()
        legs = list()
        for color, (mes, layer) in zip(colors, layers.items()):
            plots.append(ax1.barh(scales, layer, height = scales, color = color))
            legs.append(mes)
        fig.legend(plots, legs, 'lower center')
        fig.subplots_adjust(bottom=.25)
        ax1.set_xticks((-.5, .5))
        ax1.set_xticklabels(("Fail", "Success"))
        ax1.set_ylabel('scaling')
        fig.suptitle(method)




def main():
    x = np.arange(-2, 0.00001, .01)
    x0 = 6.
    #tester(fun_gen_parabola, x, x0)
    x0=3.
    x = np.arange(2, 12, .01)
    tester(fun_gen_lj, x, x0)
    plt.show()

if __name__  == '__main__':
    main()
