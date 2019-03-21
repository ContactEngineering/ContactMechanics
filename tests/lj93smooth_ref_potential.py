#
# Copyright 2019 Lars Pastewka
#           2015 Till Junge
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

## taken as reference from pycontact (https://github.com/pastewka/pycontact)

import numpy as np

def C1(epsilon, sigma, rc1, rc2):
    return epsilon/sigma*(6./5*(sigma/rc1)**10 - 3*(sigma/rc1)**4)

def C2(epsilon, sigma, rc1, rc2):
    return -12*epsilon/sigma**2*((sigma/rc1)**11 - (sigma/rc1)**5)

def C3(epsilon, sigma, rc1, rc2):
    return -(3*C1(epsilon, sigma, rc1, rc2) + \
             2*C2(epsilon, sigma, rc1, rc2)*(rc2 - rc1)) \
             /((rc2 - rc1)**2)

def C4(epsilon, sigma, rc1, rc2):
    return (2*C1(epsilon, sigma, rc1, rc2) + \
            C2(epsilon, sigma, rc1, rc2)*(rc2 - rc1)) \
            /((rc2 - rc1)**3)

def C0(epsilon, sigma, rc1, rc2):
    return C1(epsilon, sigma, rc1, rc2)*(rc2 - rc1) + \
           C2(epsilon, sigma, rc1, rc2)*(rc2 - rc1)**2/2. + \
           C3(epsilon, sigma, rc1, rc2)*(rc2 - rc1)**3/3. + \
           C4(epsilon, sigma, rc1, rc2)*(rc2 - rc1)**4/4.


###

def V(x, epsilon, sigma, rc1, rc2):
    return np.where(x < rc1,
                      epsilon*(2./15*(sigma/x)**9 - (sigma/x)**3)
                    - epsilon*(2./15*(sigma/rc1)**9 - (sigma/rc1)**3)
                    + C0(epsilon, sigma, rc1, rc2),
           np.where(x < rc2,
                      C0(epsilon, sigma, rc1, rc2)
                    - C1(epsilon, sigma, rc1, rc2)*(x - rc1)
                    - C2(epsilon, sigma, rc1, rc2)*(x - rc1)**2/2
                    - C3(epsilon, sigma, rc1, rc2)*(x - rc1)**3/3
                    - C4(epsilon, sigma, rc1, rc2)*(x - rc1)**4/4,
                    np.zeros_like(x)
                    ))


def dV(x, epsilon, sigma, rc1, rc2):
    return np.where(x < rc1,
                    - epsilon*(6./5*(sigma/x)**6 - 3)*(sigma/x)**3/x,
           np.where(x < rc2,
                    - C1(epsilon, sigma, rc1, rc2)
                    - C2(epsilon, sigma, rc1, rc2)*(x - rc1)
                    - C3(epsilon, sigma, rc1, rc2)*(x - rc1)**2
                    - C4(epsilon, sigma, rc1, rc2)*(x - rc1)**3,
                    np.zeros_like(x)
                    ))

def d2V_lj(x, epsilon, sigma, rc1, rc2):
    return 12*epsilon*((sigma/x)**6 - 1)*(sigma/x)**3/(x*x)

def d2V_smooth(x, epsilon, sigma, rc1, rc2):
    return -   C2(epsilon, sigma, rc1, rc2) \
           - 2*C3(epsilon, sigma, rc1, rc2)*(x - rc1) \
           - 3*C4(epsilon, sigma, rc1, rc2)*(x - rc1)**2

def d2V(x, epsilon, sigma, rc1, rc2):
    return np.where(x < rc1,
                    d2V_lj(x, epsilon, sigma, rc1, rc2),
           np.where(x < rc2,
                    d2V_smooth(x, epsilon, sigma, rc1, rc2),
                    np.zeros_like(x)
                    ))

###

def find_epsilon_from_gamma(gamma_target, sigma, rc1, rc2):
    return -gamma_target/V(rc1, 1.0, sigma, rc1, rc2)
