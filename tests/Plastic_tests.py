import numpy as np
import os
from scipy.optimize import bisect

from NuMPI.Tools.Reduction import Reduction
from NuMPI import MPI

from PyCo.ContactMechanics import HardWall
from PyCo.SolidMechanics import PeriodicFFTElasticHalfSpace
from PyCo.Topography import open_topography, PlasticTopography
from PyCo.System import make_system
from PyCo.Topography import Topography


FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'file_format_examples')

def test_hard_wall_bearing_area(comm, fftengine_type):
    # Test that at very low hardness we converge to (almost) the bearing
    # area geometry
    pnp = Reduction(comm)
    fullsurface = open_topography(
        os.path.join(FIXTURE_DIR, 'surface1.out')).topography()
    nb_domain_grid_pts = fullsurface.nb_grid_pts
    substrate = PeriodicFFTElasticHalfSpace(nb_domain_grid_pts, 1.0, fft="serial"
                            if comm.Get_size() == 1 else "mpi", communicator=comm)
    surface = Topography(fullsurface.heights(), physical_sizes=nb_domain_grid_pts,
                         decomposition='domain',
                         subdomain_locations=substrate.topography_subdomain_locations,
                         nb_subdomain_grid_pts=substrate.topography_nb_subdomain_grid_pts,
                         communicator=substrate.communicator)

    plastic_surface = PlasticTopography(surface, 1e-12)
    system = make_system(substrate,
                         HardWall(), plastic_surface)
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
    assert plastic_surface.plastic_area == ncontact * surface.area_per_pt
    bearing_area = bisect(lambda x: pnp.sum((surface.heights() > x)) - ncontact,
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
    system = make_system(interaction="hardwall",
                         substrate="free",
                         surface=PlasticTopography(topography=topography,
                                                   hardness=hardness),
                         young=Es,
                         communicator=comm_self
                         )

    offsets = [1e-4]
    plastic_areas = []
    contact_areas = []
    forces = np.zeros((len(offsets),
                       *topography.nb_grid_pts))  # forces[timestep,...]: array of forces for each gridpoint
    elastic_displacements = np.zeros(
        (len(offsets), *topography.nb_grid_pts))
    plastified_topographies = []

    disp0=None
    i = 0
    for offset in offsets:
        sol = system.minimize_proxy(offset=offset,
                                    # load controlled
                                    # mixfac = 1e-4,
                                    disp0=disp0,
                                    pentol=1e-10,
                                    # for the default value I had some spiky pressure fields during unloading
                                    )
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
    system = make_system(interaction="hardwall",
                         substrate="free",
                         surface=PlasticTopography(topography=topography,
                                                   hardness=hardness),
                         young=Es,
                         communicator=comm_self
                         )

    external_forces = [0.02]

    offsets = []
    plastic_areas = []
    contact_areas = []
    forces = np.zeros((len(external_forces),
                       *topography.nb_grid_pts))  # forces[timestep,...]: array of forces for each gridpoint
    elastic_displacements = np.zeros(
        (len(external_forces), *topography.nb_grid_pts))
    plastified_topographies = []

    # provide initial disp that is nonzero because otherwise the optimizer will begin with full contact area,
    # what is far from the solution in this case

    penetration = 0.00002
    disp0 = np.zeros(system.substrate.nb_domain_grid_pts)
    disp0[system.surface.subdomain_slices] = system.surface.heights() + penetration
    disp0 = np.where(disp0 > 0, disp0, 0)

    for external_force in external_forces:
        sol = system.minimize_proxy(external_force=external_force,
                                    # load controlled
                                    # mixfac = 1e-4,
                                    disp0=disp0,
                                    pentol=1e-10,
                                    # for the default value I had some spiky pressure fields during unloading
                                    )  # display informations about each iteration
        assert sol.success
        disp0 = system.disp
        offsets.append(system.offset)
        plastic_areas.append(system.surface.plastic_area)
        contact_areas.append(system.compute_contact_area())
