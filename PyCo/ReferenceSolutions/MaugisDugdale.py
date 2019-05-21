#
# Copyright 2016 Lars Pastewka
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
Maugis-Dugdale cohesive zone model for a sphere contacting an elastic
flat.
See: D. Maugis, J. Colloid Interf. Sci. 150, 243 (1992)
"""

import numpy as np
from math import pi
from scipy.optimize import brentq

def afindroot(f, left, right, a, b):
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        res = [ ]
        for ia, ib in zip(a, b):
            res += [ brentq(f, left, right, args=( ia, ib )) ]
        return np.asarray(res)
    elif isinstance(a, np.ndarray):
        res = [ ]
        for ia in a:
            res += [ brentq(f, left, right, args=( ia, b )) ]
        return np.asarray(res)
    else:
        return brentq(f, left, right, args=( a, b ))

###

def maugis_parameter(radius, elastic_modulus, work_of_adhesion,
                     cohesive_stress):
    K = 4*elastic_modulus/3
    return 2*cohesive_stress/(np.pi*work_of_adhesion*K**2/radius)**(1./3)

def fm(m, a, lam):
    mu = np.sqrt(m*m-1)
    tanmu = np.arctan(mu)
    # This is Eq. (6.17) from Maugis' paper
    return (1./2)*lam*a*a*(mu+(m*m-2)*tanmu)+(4./3)*lam*lam*a*(mu*tanmu-m+1)-1

def _load(a, lam, return_m=False):
    """
    Compute load for the Maugis-Dugdale model given the area and Maugis
    parameter.

    Parameters
    ----------
    a : array_like
        Non-dimensional contact radius
    lam : float
        Maugis parameter

    Returns
    -------
    P : array
        Non-dimensional load
    """
    m = afindroot(fm, 1.0, 1e12, a, lam)
    mu = np.sqrt(m*m-1)
    tanmu = np.arctan(mu)
    P = a*a*a-lam*a*a*(mu+m*m*tanmu)
    if return_m:
        return P, m
    else:
        return P

def _load_and_displacement(a, lam, return_m=False):
    """
    Compute load and displacement for the Maugis-Dugdale model given the area
    and Maugis parameter.

    Parameters
    ----------
    a : array_like
        Non-dimensional contact radius
    lam : float
        Maugis parameter

    Returns
    -------
    N : array
        Non-dimensional load
    d : array
        Non-dimensional displacement
    """
    P, m = _load(a, lam, return_m=True)
    mu = np.sqrt(m*m-1)
    d = a*a-(4./3)*a*lam*mu
    if return_m:
        return P, d, m
    else:
        return P, d

def load_and_displacement(contact_radius, radius, elastic_modulus,
                          work_of_adhesion, cohesive_stress):
    lam = maugis_parameter(radius, elastic_modulus, work_of_adhesion,
                          cohesive_stress)
    K = 4*elastic_modulus/3
    A = contact_radius/(np.pi*work_of_adhesion*radius**2/K)**(1./3)
    N, d = _load_and_displacement(A, lam)
    return (N*np.pi*work_of_adhesion*radius,
            d*(np.pi**2*work_of_adhesion**2*radius/K**2)**(1./3))
