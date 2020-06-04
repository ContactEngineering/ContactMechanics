#
# Copyright 2017, 2020 Lars Pastewka
#           2018, 2020 Antoine Sanner
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
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

import numpy as np
import os
from scipy.optimize import bisect

from NuMPI.Tools.Reduction import Reduction

from ContactMechanics import PeriodicFFTElasticHalfSpace
from SurfaceTopography import open_topography, PlasticTopography
from ContactMechanics.PlasticSystemSpecialisations import \
    PlasticNonSmoothContactSystem
from ContactMechanics.Factory import make_plastic_system
from SurfaceTopography import Topography

FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '.')


def test_hard_wall_bearing_area(comm):
    # Test that at very low hardness we converge to (almost) the bearing
    # area geometry
    pnp = Reduction(comm)
    fullsurface = open_topography(
        os.path.join(FIXTURE_DIR, 'surface1.out')).topography()
    nb_domain_grid_pts = fullsurface.nb_grid_pts
    substrate = PeriodicFFTElasticHalfSpace(nb_domain_grid_pts, 1.0, fft='mpi',
                                            communicator=comm)
    surface = Topography(
        fullsurface.heights(),
        physical_sizes=nb_domain_grid_pts,
        decomposition='domain',
        subdomain_locations=substrate.topography_subdomain_locations,
        nb_subdomain_grid_pts=substrate.topography_nb_subdomain_grid_pts,
        communicator=substrate.communicator)

    plastic_surface = PlasticTopography(surface, 1e-12)
    system = PlasticNonSmoothContactSystem(substrate, plastic_surface)
    offset = -0.002
    if comm.rank == 0:
        def cb(it, p_r, d):
            print("{0}: area = {1}".format(it, d["area"]))
    else:
        def cb(it, p_r, d):
            pass

    result = system.minimize_proxy(offset=offset, callback=cb)
    assert result.success
    c = result.jac > 0.0
    ncontact = pnp.sum(c)
    assert plastic_surface.plastic_area == ncontact * surface.area_per_pt
    bearing_area = bisect(
        lambda x: pnp.sum((surface.heights() > x)) - ncontact,
        -0.03, 0.03)
    cba = surface.heights() > bearing_area
    # print(comm.Get_rank())
    assert pnp.sum(np.logical_not(c == cba)) < 25


def test_hardwall_plastic_nonperiodic_disp_control(comm_self):
    # test just that it works without bug, not accuracy

    nx, ny = 128, 128

    sx = 0.005  # mm
    sy = 0.005  # mm

    x = np.arange(0, nx).reshape(-1, 1) * sx / nx - sx / 2
    y = np.arange(0, ny).reshape(1, -1) * sy / ny - sy / 2

    topography = Topography(- np.sqrt(x ** 2 + y ** 2) * 0.05,
                            physical_sizes=(sx, sy))

    Es = 230000  # MPa
    hardness = 6000  # MPa
    system = make_plastic_system(substrate="free",
                                 surface=PlasticTopography(
                                     topography=topography,
                                     hardness=hardness),
                                 young=Es,
                                 communicator=comm_self
                                 )

    offsets = [1e-4]

    disp0 = None
    for offset in offsets:
        sol = system.minimize_proxy(offset=offset,
                                    disp0=disp0,
                                    pentol=1e-10,)
        assert sol.success


def test_hardwall_plastic_nonperiodic_load_control(comm_self):
    # test just that it works without bug, not accuracy

    nx, ny = 128, 128

    sx = 0.005  # mm
    sy = 0.005  # mm

    x = np.arange(0, nx).reshape(-1, 1) * sx / nx - sx / 2
    y = np.arange(0, ny).reshape(1, -1) * sy / ny - sy / 2

    topography = Topography(- np.sqrt(x ** 2 + y ** 2) * 0.05,
                            physical_sizes=(sx, sy))

    Es = 230000  # MPa
    hardness = 6000  # MPa
    system = make_plastic_system(substrate="free",
                                 surface=PlasticTopography(
                                     topography=topography,
                                     hardness=hardness),
                                 young=Es,
                                 communicator=comm_self
                                 )

    external_forces = [0.02]

    offsets = []
    plastic_areas = []
    contact_areas = []

    # provide initial disp that is nonzero because otherwise the optimizer will
    # begin with full contact area, what is far from the solution in this case

    penetration = 0.00002
    disp0 = np.zeros(system.substrate.nb_domain_grid_pts)
    disp0[system.surface.subdomain_slices] = \
        system.surface.heights() + penetration
    disp0 = np.where(disp0 > 0, disp0, 0)

    for external_force in external_forces:
        sol = system.minimize_proxy(external_force=external_force,
                                    disp0=disp0,
                                    pentol=1e-10)
        assert sol.success
        disp0 = system.disp
        offsets.append(system.offset)
        plastic_areas.append(system.surface.plastic_area)
        contact_areas.append(system.compute_contact_area())
