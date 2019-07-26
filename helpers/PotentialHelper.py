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
symbolic calculations for potentials
"""

from sympy import Symbol, pprint
import sympy


r = Symbol('r', positive=True)

def show_props(pot, name):
    print("{} potential:".format(name))
    #dpot = sympy.simplify(sympy.diff(pot, r))
    #ddpot = sympy.simplify(sympy.diff(dpot, r))
    dpot = sympy.diff(pot, r)
    ddpot = sympy.diff(dpot, r)
    pprint(pot)
    print("1st derivative:")
    pprint(dpot)
    print("2nd derivative:")
    pprint(ddpot)

    r_min = sympy.solve(dpot, r)
    print('r_min =')
    pprint(r_min[-1])

    r_infl = sympy.solve(ddpot, r)
    print('r_infl =')
    pprint(r_infl[-1])
    print()


def main():
    eps = Symbol('ε', positive=True)
    sig = Symbol('σ', positive=True)

    LJ93 = eps*(-(sig/r)**3 + (2*sig**9)/(15*r**9))
    show_props(LJ93, 'LJ')

    hama = Symbol("A", positive=True)
    c_sr = Symbol("C_sr", positive=True)
    vdW82 = -hama/(12*sympy.pi*r**2) + c_sr*r**-8
    show_props(vdW82, 'LJ')

if __name__ == '__main__':
    main()
