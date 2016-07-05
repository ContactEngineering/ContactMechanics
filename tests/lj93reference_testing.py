#!/usr/bin/env python

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
