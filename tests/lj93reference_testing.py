#
# Copyright 2016 Lars Pastewka
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

from . import lj93_ref_potential as lj93
from . import lj93smooth_ref_potential as lj93s
import numpy as np

eps = 1.
sig = 1.
rc1 = 2.5
rc2 = 1.



xmin, xmax, step = (.7, 3, .01)
x = np.arange(xmin, xmax, step)

v  = lj93. V(x, eps, sig, rc1)
vs = lj93s.V(x, eps, sig, 1.1, rc1)

#import matplotlib.pyplot as plt
## plt.plot(x, v,  label='std')
## plt.plot(x, vs, label='smooth')
## plt.legend(loc='best')
## plt.show()
