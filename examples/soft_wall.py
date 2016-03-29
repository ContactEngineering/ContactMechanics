#! /usr/bin/env python3

"""
Automatically compute contact area, load and displacement for a rough surface.
Tries to guess displacements such that areas are equally spaced on a log scale.
"""

import numpy as np
from PyCo.ContactMechanics import LJ93
from PyCo.SolidMechanics import PeriodicFFTElasticHalfSpace
from PyCo.Surface import read_matrix
from PyCo.System import SystemFactory
from PyCo.Tools.Logger import Logger, quiet, screen
from PyCo.Tools.NetCDF import NetCDFContainer

###

# Total number of area/load/displacements to use
nsteps = 20

# Maximum number of iterations per data point
maxiter = 30

###

def next_step(system, surface, history=None, r_min=0.0, logger=quiet):
    """
    Run a full contact calculation. Try to guess displacement such that areas
    are equally spaced on a log scale.

    Parameters
    ----------
    system : PyCo.System.SystemBase object
        The contact mechanical system.
    surface : PyCo.Surface.Surface object
        The rigid rough surface.
    history : tuple
        History returned by past calls to next_step 

    Returns
    -------
    displacements : numpy.ndarray
        Current surface displacement field.
    forces : numpy.ndarray
        Current surface pressure field.
    displacement : float
        Current displacement of the rigid surface
    load : float
        Current load.
    area : float
        Current fractional contact area.
    history : tuple
        History of contact calculations.
    """

    # Get the profile as a numpy array
    profile = surface.profile()

    # Find max, min and mean heights
    top = np.max(profile)
    middle = np.mean(profile)
    bot = np.min(profile)

    if history is None:
        step = 0
    else:
        disp, gap, load, area, converged = history
        step = len(disp)

    if step == 0:
        disp = []
        gap = []
        load = []
        area = []
        converged = np.array([], dtype=bool)

        disp0 = -middle+r_min
    elif step == 1:
        disp0 = -top+0.01*(top-middle)+r_min
    else:
        ref_area = np.log10(np.array(area+1/np.prod(surface.shape)))
        darea = np.append(ref_area[1:]-ref_area[:-1], -ref_area[-1])
        i = np.argmax(darea)
        if i == step-1:
            disp0 = bot+r_min+2*(disp[-1]-bot-r_min)
        else:
            disp0 = (disp[i]+disp[i+1])/2

    opt = system.minimize_proxy(disp0, method='CG', tol=1.0)
    u = opt.x
    f = opt.jac
    # minimize_proxy returns a raveled array
    u.shape = profile.shape
    f.shape = profile.shape
    disp = np.append(disp, [disp0])
    gap = np.append(gap, [np.mean(u)-middle-disp0])
    current_load = f.sum()/np.prod(surface.size)
    load = np.append(load, [current_load])
    current_area = (f>0).sum()/np.prod(surface.shape)
    area = np.append(area, [current_area])
    converged = np.append(converged, np.array([opt.success], dtype=bool))
    logger.pr('disp = {}, area = {}, load = {}, converged = {}' \
        .format(disp0, current_area, current_load, opt.success))
    if not opt.success:
        logger.pr(opt.message)

    # Sort by area
    disp, gap, load, area, converged = np.transpose(sorted(zip(disp, gap, load,
                                                               area, converged),
                                                    key=lambda x: x[3]))
    converged = np.array(converged, dtype=bool)

    return u, f, disp0, current_load, current_area, \
        (disp, gap, load, area, converged)

###

# Read a surface topography from a text file. Returns a PyCo.Surface.Surface
# object.
surface = read_matrix('surface1.out', factor=1000)
# Set the *physical* size of the surface. We here set it to equal the shape,
# i.e. the resolution of the surface just read. Size is returned by surface.size
# and can be unknown, i.e. *None*.
surface.set_size(surface.shape)

# Initialize elastic half-space. This one is periodic with contact modulus
# E*=1.0 and physical size equal to the surface.
substrate = PeriodicFFTElasticHalfSpace(surface.shape, 1.0, surface.size)
# Soft-wall interaction. This is a 9-3 Lennard-Jones potential.
interaction = LJ93(1.0, 1.0)
print(interaction.r_min, surface.compute_rms_height())
# Cut it off at the minimum
interaction = LJ93(1.0, 1.0, r_cut=interaction.r_min)
# Piece the full system together. In particular the PyCo.System.SystemBase
# object knows how to optimize the problem. For the hard wall interaction it
# will always use Polonsky & Keer's constrained conjugate gradient method.
system = SystemFactory(substrate, interaction, surface)

###

# Create a NetCDF container to dump displacements and forces to.
container = NetCDFContainer('traj.nc', mode='w', double=True)
container.set_shape(surface.shape)

# Additional log file for load and area
txt = Logger('soft_wall.out')

history = None
for i in range(nsteps):
    displacements, forces, disp0, load, area, history = \
        next_step(system, surface, history, r_min=interaction.r_min,
                  logger=screen)
    frame = container.get_next_frame()
    frame.displacements = displacements
    frame.forces = forces
    frame.displacement = disp0
    frame.load = load
    frame.area = area

    txt.st(['displacement', 'load', 'area'],
           [disp0, load, area])

container.close()