#
# Copyright 2018-2019 Antoine Sanner
#           2019 Lars Pastewka
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
Tests plastic deformation
"""
try:
    import numpy as np
    import pytest
    from scipy.optimize import bisect
    from PyCo.ContactMechanics import HardWall
    from PyCo.SolidMechanics import PeriodicFFTElasticHalfSpace
    from PyCo.Topography import open_topography, PlasticTopography
    from PyCo.System import make_system
    from NuMPI.Tools.Reduction import Reduction

    from PyCo.Topography import Topography
    from runtests.mpi import MPITest
    import os
    import PyCo
except ImportError as err:
    import sys

    print(err)
    sys.exit(-1)
FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'file_format_examples')

def test_hard_wall_bearing_area(comm, fftengine_type):
    # Test that at very low hardness we converge to (almost) the bearing
    # area geometry
    pnp = Reduction(comm)
    fullsurface = open_topography(os.path.join(FIXTURE_DIR, 'surface1.out')).topography()
    nb_domain_grid_pts = fullsurface.nb_grid_pts
    substrate = PeriodicFFTElasticHalfSpace(nb_domain_grid_pts, 1.0, fft="mpi",comm=comm)
    surface = Topography(fullsurface.heights(), physical_sizes=nb_domain_grid_pts,
                         subdomain_locations=substrate.topography_subdomain_locations,
                         nb_subdomain_grid_pts=substrate.topography_nb_subdomain_grid_pts,
                         pnp=substrate.pnp)

    system = make_system(substrate,
                         HardWall(), PlasticTopography(surface, 0.0000000001))
    offset = -0.002
    if comm.rank == 0:
        def cb(it, p_r, d):
            print("{0}: area = {1}".format(it, d["area"]))
    else:
        def cb(it, p_r, d):
            pass

    result = system.minimize_proxy(offset=offset,  callback=cb)
    assert result.success
    c = result.jac > 0.0
    ncontact = pnp.sum(c)
    bearing_area = bisect(lambda x: pnp.sum((surface.heights() > x)) - ncontact,
                          -0.03, 0.03)
    cba = surface.heights() > bearing_area
    # print(comm.Get_rank())
    assert pnp.sum(np.logical_not(c == cba)) < 25
