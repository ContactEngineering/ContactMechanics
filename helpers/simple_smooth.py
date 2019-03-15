#
# Copyright 2017, 2019 Lars Pastewka
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
simplified smoothing for potentials
"""

import numpy as np
from PyCo.ContactMechanics import VDW82


import matplotlib.pyplot as plt

hamaker = 68.1e-21
c_sr = 2.1e-78*1e-6
print(c_sr)

pot = VDW82(c_sr, hamaker)

r_c = 5e-10
r = np.linspace(.7*pot.r_min, 2*r_c, 1000)

V, forces, ddV = pot.evaluate(r, pot=True, forces=True, curb=True)
dV = -forces

deltaV, forces, deltaddV = [-float(bla) for bla in pot.evaluate(r_c, pot=True, forces=True, curb=True)]
deltadV = -forces

deltadV -= deltaddV*r_c

deltaV -= deltaddV/2*r_c**2 + deltadV*r_c

print("a, b, c = ({0})".format((deltaV, deltadV, deltaddV)))



fig = plt.figure()

ax = fig.add_subplot(311)
ax.grid(True)
plt.plot(r, V, label='original')
V1 = V + deltaddV/2*r**2
plt.plot(r, V1, label='step 1')
V2 = V1 + deltadV*r
plt.plot(r, V2, label='step 2')
V3 = V2 + deltaV
plt.plot(r, V3, label='step 3')
ax.set_ylim(bottom=1.1*V2.min(), top = -1.1*V2.min())
ax.legend(loc='best')

ax = fig.add_subplot(312)
ax.grid(True)
plt.plot(r, dV)
dV1 = dV + deltaddV*r
plt.plot(r, dV1)
dV2 = dV1 + deltadV
plt.plot(r, dV2)
ax.set_ylim(bottom=-1.1*dV1.max(), top = 1.1*dV1.max())

ax = fig.add_subplot(313)
ax.grid(True)
plt.plot(r, ddV)
plt.plot(r, ddV+deltaddV)
ax.set_ylim(bottom=1.1*ddV.min(), top = -1.1*ddV.min())


plt.show()
