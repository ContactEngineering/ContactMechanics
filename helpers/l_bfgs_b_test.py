#
# Copyright 2015-2016 Till Junge
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
sample problem to get a grasp on how te precondition equaitons for
scipy's l-bfgs-b solver
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
