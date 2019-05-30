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
Uses cProfile and Snakeviz to show in which function the most time is spent
"""


import numpy as np
from PyCo.SolidMechanics import FreeFFTElasticHalfSpace
from PyCo.Topography import make_sphere

from FFTEngine import PFFTEngine
from NuMPI.Optimization import LBFGS
from NuMPI.Tools.Reduction import Reduction


from PyCo.ContactMechanics import VDW82smoothMin
from PyCo.System import SmoothContactSystem


from NuMPI.IO import save_npy
from NuMPI import MPI


import cProfile

def profile(filename=None, comm=MPI.COMM_WORLD):
  def prof_decorator(f):
    def wrap_f(*args, **kwargs):
      pr = cProfile.Profile()
      pr.enable()
      result = f(*args, **kwargs)
      pr.disable()

      if filename is None:
        pr.print_stats()
      else:
        filename_r = filename + ".{}".format(comm.rank)
        pr.dump_stats(filename_r)

      return result
    return wrap_f
  return prof_decorator


comm = MPI.COMM_WORLD

# sphere radius:
r_s = 10.0
# contact radius
r_c = .2
# peak pressure
p_0 = 2.5
# equivalent Young's modulus
E_s = 102.#102.
# work of adhesion
w = 1.0
# tolerance for optimizer
tol = 1e-12
# tolerance for contact area
gap_tol = 1e-6

n = 512 * comm.Get_size()

nx, ny = n, n
sx = 21.0

z0 = 0.05 # needed to get small tolerance, but very very slow

fftengine = PFFTEngine((2*nx, 2*ny), comm=comm)
pnp =Reduction(comm=comm)

# the "Min" part of the potential (linear for small z) is needed for the LBFGS without bounds
inter = VDW82smoothMin(w * z0 ** 8 / 3, 16 * np.pi * w * z0 ** 2, gamma=w, pnp = pnp)

# Parallel Topography Patch

substrate = FreeFFTElasticHalfSpace((nx,ny), young=E_s, physical_sizes=(sx, sx), fft=fftengine, pnp=pnp)
print(substrate._comp_nb_grid_pts)
print(fftengine.nb_domain_grid_pts)


surface = make_sphere(radius=r_s, nb_grid_pts=(nx, ny), size=(sx, sx),
                      subdomain_locations=substrate.topography_subdomain_locations,
                      nb_subdomain_grid_pts=substrate.topography_nb_subdomain_grid_pts,
                      pnp=pnp,
                      standoff=float('inf'))
ext_surface = make_sphere(r_s, (2 * nx, 2 * ny), (2 * sx, 2 * sx),
                          centre=(sx / 2, sx / 2),
                          subdomain_locations=substrate.subdomain_locations,
                          nb_subdomain_grid_pts=substrate.nb_subdomain_grid_pts,
                          pnp=pnp,
                          standoff=float('inf'))
system = SmoothContactSystem(substrate, inter, surface)

penetration = 0

disp0 = ext_surface.heights() + penetration
disp0 = np.where(disp0 > 0, disp0, 0)
disp0 = system.shape_minimisation_input(disp0)

def do_step():

    result = LBFGS(system.objective(penetration, gradient=True), disp0, jac=True,
                   pnp=pnp, gtol=1e-6 * abs(w/z0), ftol=1e-20)


    #result = system.minimize_proxy(offsets[i], disp0=None,method = LBFGS,options=dict(gtol = 1e-3, maxiter =100,maxls=10))

    u = result.x
    u.shape = ext_surface.nb_subdomain_grid_pts
    f = substrate.evaluate_force(u)
    converged = result.success
    assert converged

    gap = system.compute_gap(u, penetration)

    save_npy("gap_profiling.npy",
             gap[tuple([slice(None, r) for r in
                        substrate.topography_nb_subdomain_grid_pts])],
             substrate.topography_subdomain_locations,
             substrate.nb_grid_pts,
             comm=comm)


do_step=profile("profile_out_{}procs".format(comm.Get_size()), comm)(do_step)
do_step()

# then call snakeviz profile_out.<rank> to get the see the performance analysis

import subprocess
subprocess.call("snakeviz {}".format("profile_out_{}procs.{}".format(comm.Get_size(), comm.Get_rank())), shell=True)
