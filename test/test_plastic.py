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

from SurfaceTopography import open_topography, PlasticTopography
from SurfaceTopography import Topography, read_published_container

from ContactMechanics import PeriodicFFTElasticHalfSpace
from ContactMechanics.PlasticSystemSpecialisations import \
    PlasticNonSmoothContactSystem
from ContactMechanics.Factory import make_plastic_system

FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '.')


def test_hard_wall_bearing_area(comm):
    # Test that at very low hardness we converge to (almost) the bearing
    # area geometry
    pnp = Reduction(comm)
    fullsurface = open_topography(
        os.path.join(FIXTURE_DIR, 'surface1.out'))
    fullsurface = fullsurface.topography(physical_sizes=fullsurface.channels[0].nb_grid_pts)
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
            # print("{0}: area = {1}".format(it, d["area"]))
            pass
    else:
        def cb(it, p_r, d):
            pass

    result = system.minimize_proxy(offset=offset, callback=cb, maxiter=1000)
    assert result.success, 'Minimization did not succeed'
    ncontact = pnp.sum(result.active_set)
    bearing_area = bisect(lambda x: pnp.sum((surface.heights() > x)) - ncontact, -0.03, 0.03)
    cba = surface.heights() > bearing_area
    assert plastic_surface.plastic_area == ncontact * surface.area_per_pt, 'The full contact should be plastic'
    assert pnp.sum(np.logical_not(result.active_set == cba)) < 25, 'Contact area is not identical to the bearing area'


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
        sol = system.minimize_proxy(offset=offset, initial_displacements=disp0,
                                    pentol=1e-10)
        assert sol.success, 'Minimization did not succeed'


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
    system = make_plastic_system(
        substrate="free",
        surface=PlasticTopography(topography=topography, hardness=hardness),
        young=Es,
        communicator=comm_self)

    external_forces = [0.02, 0.06, 0.12]

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
        sol = system.minimize_proxy(
            external_force=external_force,
            initial_displacements=disp0,
            pentol=1e-10,
            maxiter=100)
        assert sol.success, 'Minimization did not succeed'
        disp0 = system.disp
        offsets.append(system.offset)
        plastic_areas.append(system.surface.plastic_area)
        contact_areas.append(system.compute_contact_area())

        assert np.max(system.force / system.area_per_pt) < hardness, 'Maximmum force exceeded hardnees'


def test_automatic_offsets(comm_self):
    c, = read_published_container('https://contact.engineering/go/867nv')
    t = c[2]
    assert t.info['name'] == '500x500_random.txt', 'Topography has wrong name'

    def check_result(displacement_xy, gap_xy, pressure_xy, contacting_points_xy, mean_displacement, mean_pressure,
                     total_contact_area):
        if total_contact_area > 0:
            assert np.any((t.heights() + mean_displacement) > 0), 'Contact area should be zero but is not'
            np.testing.assert_array_less(0.0, displacement_xy)
        else:
            np.testing.assert_array_less(t.heights() + mean_displacement, 0)  # Check that there should be no contact
            np.testing.assert_array_almost_equal(displacement_xy, 0.0)

    # Elastic
    dois = set()
    t.contact_mechanics(nsteps=10, results_callback=check_result, dois=dois)
    assert dois == {
        '10.1115/1.2833523',  # Stanley & Kato
        '10.1103/PhysRevB.74.075420',  # Campana, Müser
        '10.1103/PhysRevB.86.075459',  # Pastewka, Sharp, Robbins
        '10.1016/S0043-1648(99)00113-1',  # Polonsky, Keer
        'Hockney, Methods Comput. Phys. 9, 135 (1970)',
        '10.1016/S0043-1648(00)00427-0',  # Liu, Wang, Liu 200
        '10.1063/1.4950802'  # Pastewka, Robbins 2016
    }, 'Contact mechanics calculation returned wrong list of references'

    # Plastic (should reset plasticity before every step, otherwise above idiot check will fail)
    dois = set()
    t.contact_mechanics(nsteps=10, hardness=0.05, results_callback=check_result, dois=dois)
    assert dois == {
        '10.1115/1.2833523',  # Stanley & Kato
        '10.1103/PhysRevB.74.075420',  # Campana, Müser
        '10.1103/PhysRevB.86.075459',  # Pastewka, Sharp, Robbins
        '10.1016/S0043-1648(99)00113-1',  # Polonsky, Keer
        'Hockney, Methods Comput. Phys. 9, 135 (1970)',
        '10.1016/S0043-1648(00)00427-0',  # Liu, Wang, Liu 200
        '10.1063/1.4950802',  # Pastewka, Robbins 2016
        '10.1016/j.triboint.2005.11.008',  # Almqvist et al. 2007
        '10.1038/s41467-018-02981-y'  # Weber et al. 2018
    }, 'Contact mechanics calculation returned wrong list of references'
