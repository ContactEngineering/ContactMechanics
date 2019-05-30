#
# Copyright 2019 Antoine Sanner
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
I should use better units, there are often abnormal termination in linesearch problems

"""


import time
import os
import scipy.optimize
import numpy as np

from PyCo.SolidMechanics import PeriodicFFTElasticHalfSpace
from PyCo.Topography.Generation import fourier_synthesis

from FFTEngine import PFFTEngine
from NuMPI.Optimization import LBFGS
from NuMPI.Tools.Reduction import Reduction
from PyCo.ContactMechanics import VDW82smoothMin
from PyCo.System import SmoothContactSystem


from NuMPI.IO import save_npy
from NuMPI import MPI

import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
pnp = Reduction(comm=comm)

class iter_inspector():
    def __init__(self):
        self.neval = 0
        self.energies = []
        self.maxgradients =[]

    def __call__(self, system):
        self.neval += 1
        self.energies.append(system.energy)
        self.maxgradients.append(pnp.max(system.force))


class decorated_objective:
    def __init__(self, system, objective):
        self.system = system
        self.objective = objective
        self.neval = 0
        self.energies = []
        self.maxgradients = []

    def __call__(self, *args, **kwargs):
        val = self.objective(*args, **kwargs)
        self.neval += 1
        self.energies.append(system.energy)
        self.maxgradients.append(pnp.max(abs(system.force)))
        return val


import matplotlib.pyplot as plt


fig, (axEn, axgrad) = plt.subplots(2,1, sharex = True)
n = 512

axEn.set_xlabel("# of objective evaluations")
axEn.set_ylabel("E(i)-E(last) / (E(0)-E(last))")
axEn.set_yscale("log")

axgrad.set_yscale("log")
axgrad.set_ylabel(r"$|grad|_{\infty}$")


for a in (axEn, axgrad):
    a.set_xlabel("# of objective evaluations")
    a.label_outer()
for method, name in zip(["L-BFGS-B", LBFGS],
                        ["Scipy", "NuMPI"]):

    print("#################")
    print(name)
    # equivalent Young's modulus
    E_s = 1000000.  # 102.
    # work of adhesion

    nx, ny = n, n
    dx = 1.0
    dy = 1.0 # in units of z0
    sx = nx * dx
    sy = ny * dy

    z0 = 4.0  # needed to get small tolerance, but very very slow

    w = 0.01 * E_s * z0

    fftengine = PFFTEngine((nx, ny), comm=comm)

    # the "Min" part of the potential (linear for small z) is needed for the LBFGS without bounds
    inter = VDW82smoothMin(w * z0 ** 8 / 3, 16 * np.pi * w * z0 ** 2,
                           gamma=w, pnp=pnp)

    # Parallel Topography Patch

    substrate = PeriodicFFTElasticHalfSpace((nx, ny), young=E_s,
                                            physical_sizes=(sx, sx), pnp=pnp)
    # print(substrate._comp_nb_grid_pts)
    # print(fftengine.nb_domain_grid_pts)


    surface = fourier_synthesis((nx, ny), (sx, sy), hurst=0.8, rms_height=1, short_cutoff=8, long_cutoff=sx / 2)

    system = SmoothContactSystem(substrate, inter, surface)

    penetration = -0.1

    disp0 = surface.heights() + penetration
    disp0 = np.where(disp0 > 0, disp0, 0)
    # disp0 = system.shape_minimisation_input(disp0)

    maxcor = 10

    starttime = time.time()

    objective_monitor = decorated_objective(system, system.objective(
        penetration, gradient=True))
    try:
        result = scipy.optimize.minimize(objective_monitor,
                                         disp0, method=method, jac=True,
                                         options=dict(
                                             gtol=1e-6 * abs(w / z0),
                                             ftol=1e-30,
                                             maxcor=maxcor))
        converged = result.success
        assert converged
    except Exception as err:
        print("went wrong")
        print(err)

    print(method)
    print(result.message)
    print("nevals: {}".format(objective_monitor.neval))
    print(result.nit)

    axgrad.plot(range(objective_monitor.neval), objective_monitor.maxgradients, label="{}".format(name))
    axEn.plot(range(objective_monitor.neval), (objective_monitor.energies - objective_monitor.energies[-1] )/ (objective_monitor.energies[0] - objective_monitor.energies[-1]), label="{}".format(name))


axgrad.legend()
fig.suptitle("n={}".format(n))
fig.savefig("{}.png".format(os.path.basename(__file__)))


