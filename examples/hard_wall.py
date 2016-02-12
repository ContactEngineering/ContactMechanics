#! /usr/bin/env python3

"""
Automatically compute contact area, load and displacement for a rough surface.
Tries to guess displacements such that areas are equally spaced on a log scale.
"""

import numpy as np
from PyCo.ContactMechanics import HardWall
from PyCo.SolidMechanics import PeriodicFFTElasticHalfSpace
from PyCo.Surface import read_asc
from PyCo.System import SystemFactory
from PyCo.Tools.Logger import screen
from PyCo.Tools.NetCDF import NetCDFContainer

###

# Total number of area/load/displacements to use
nsteps = 20

# Maximum number of iterations per data point
maxiter = 30

###

def next_step(system, surface, history=None):
    """
    Run a full contact calculation. Try to guess displacement such that areas
    are equally spaced on a log scale.
    """

    profile = surface.profile()
    top = np.max(profile)
    middle = np.mean(profile)
    bot = np.min(profile)

    if history is None:
        step = 0
    else:
        disp, load, area, gap, converged = history
        step = len(disp)

    if step == 0:
        disp = []
        gap = []
        load = []
        area = []
        converged = np.array([], dtype=bool)

        disp0 = -middle
    elif step == 1:
        disp0 = -top+0.01*(top-middle)
    else:
        ref_area = np.log10(np.array(area))
        darea = np.append(ref_area[1:]-ref_area[:-1], -ref_area[-1])
        i = np.argmax(darea)
        if i == step-1:
            disp0 = bot+2*(disp[-1]-bot)
        else:
            disp0 = (disp[i]+disp[i+1])/2

    opt = system.minimize_proxy(disp0, pentol=1e-3, maxiter=maxiter,
                                logger=screen)
    u = opt.x
    f = opt.jac
    c = opt.success
    disp = np.append(disp, [disp0])
    gap = np.append(gap, [np.mean(u)-middle-disp0])
    current_load = f.sum()/np.prod(surface.size)
    load = np.append(load, [current_load])
    current_area = (f>0).sum()/np.prod(surface.shape)
    area = np.append(area, [current_area])
    converged = np.append(converged, np.array([c], dtype=bool))

    # Sort by area
    disp, gap, load, area, converged = np.transpose(sorted(zip(disp, gap, load,
                                                               area, converged),
                                                    key=lambda x: x[3]))

    converged = np.array(converged, dtype=bool)

    return u, f, disp0, current_load, current_area, \
        (disp, gap, load, area, converged)

###

surface = read_asc('surface1.out')
surface.set_size(surface.shape)

substrate = PeriodicFFTElasticHalfSpace(surface.shape, 1.0, surface.size)
interaction = HardWall()
system = SystemFactory(substrate, interaction, surface)

###

container = NetCDFContainer('traj.nc', mode='w', double=True)
container.set_shape(surface.shape)

history = None
for i in range(nsteps):
    displacements, forces, disp0, load, area, history = \
        next_step(system, surface, history)
    frame = container.get_next_frame()
    frame.displacements = displacements
    frame.forces = forces
    frame.displacement = disp0
    frame.load = load
    frame.area = area

container.close()