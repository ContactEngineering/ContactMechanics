
# coding: utf-8

## Testing the Augmented Lagrangian of PyCo

# The implementation of the augmented Lagrangian in PyCo.Tools follows closely the description of the `LANCELOT` algorithm described in Bierlaire (2006)

# The function `augmented_lagrangian` has the form of custom minimizer for [scipy.optimize.minimize](http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.optimize.minimize.html)

# In[4]:

import sys
import os
import numpy as np

import scipy.optimize
sys.path.append(os.path.join(os.getcwd(), "../PyCo/Tools/"))
from AugmentedLagrangian import augmented_lagrangian


### Book example

# Example 20.5: Minimise the fuction $f(x)$
# $$\min_{x\in\mathbb{R}^2} 2(x_1^2+x_2^2 -1)-x_1$$
# under the constraint
# $$ x_1^2 + x_2^2 = 1$$

# ugly workaround to get a fresh AugmentedLagrangian without module loads

# In[9]:

# fname = "../PyCo/Tools/AugmentedLagrangian.py"
# with open(fname) as filehandle:
#     content = ''.join((line for line in filehandle))
# exec(content)


# In[11]:

def fun(x):
    return (x[0]**2 + x[1]**2 - 1) - x[0]
def constraint(x):
    return x[0]**2 + x[1]**2 - 1
tol = 1.e-2
result = scipy.optimize.minimize(fun, x0=np.array((-1, .1)),
       	                         constraints={'type':'eq','fun':constraint},
	                         method=augmented_lagrangian, tol=tol,
	                         options={'multiplier0': np.array((0.)),
                                          'disp': True,
                                          'store_iterates': 'iterate'})

print(result)

