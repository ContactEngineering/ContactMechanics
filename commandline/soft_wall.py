#! /usr/bin/env python3

"""
Automatically compute contact area, load and displacement for a rough surface.
Tries to guess displacements such that areas are equally spaced on a log scale.
"""

import numpy as np
from PyCo.ContactMechanics import ExpPotential, LJ93smoothMin
from PyCo.SolidMechanics import PeriodicFFTElasticHalfSpace
from PyCo.Topography import read_matrix
from PyCo.System import SystemFactory
from PyCo.Tools.Logger import Logger, quiet, screen
from PyCo.Tools.NetCDF import NetCDFContainer

###

# Total number of area/load/displacements to use
nsteps = 4

# Maximum number of iterations per data point
maxiter = 1000

#gamma = 2e-5
#rho = 2.071e-5
gamma = 0.01
rho = 1.0

###

# Read a surface topography from a text file. Returns a PyCo.Topography.Topography
# object.
surface = read_matrix('surface1.out')
# Set the *physical* size of the surface. We here set it to equal the shape,
# i.e. the resolution of the surface just read. Size is returned by surface.size
# and can be unknown, i.e. *None*.
surface.set_size(surface.shape)

# Initialize elastic half-space. This one is periodic with contact modulus
# E*=1.0 and physical size equal to the surface.
substrate = PeriodicFFTElasticHalfSpace(surface.shape, 1.0, surface.size,
                                        stiffness_q0=None)

interaction = ExpPotential(gamma, rho)
#interaction = LJ93smoothMin(1.0, 1.0)

# Piece the full system together. In particular the PyCo.System.SystemBase
# object knows how to optimize the problem. For the hard wall interaction it
# will always use Polonsky & Keer's constrained conjugate gradient method.
system = SystemFactory(substrate, interaction, surface)

###

# Create a NetCDF container to dump displacements and forces to.
container = NetCDFContainer('traj_s.nc', mode='w', double=True)
container.set_shape(surface.shape)

u = None
tol = 1e-9
for disp0 in np.linspace(-10, 10, 11):
    opt = system.minimize_proxy(disp0, u, lbounds=surface.profile()+disp0,
                                method='L-BFGS-B', tol=0.0001)
    #opt = system.minimize_proxy(disp0, x0, method='L-BFGS-B', tol=0.0001)
    u = opt.x
    # minimize_proxy returns a raveled array
    u.shape = surface.shape
    # We need to reevaluate the force. What the (bounded) optimizer returns as
    # a jacobian is NOT the force.
    f = substrate.evaluate_force(u)

    gap = u-surface.profile()-disp0
    mean_gap = np.mean(gap)
    load = -f.sum()/np.prod(surface.size)
    #area = (f>0).sum()/np.prod(surface.shape)
    area = (gap<tol).sum()/np.prod(surface.shape)
    if not opt.success:
        screen.pr('Minimization failed: {}'.format(opt.message))

    frame = container.get_next_frame()
    frame.displacements = u
    frame.gap = gap
    frame.forces = f
    frame.displacement = disp0
    frame.load = load
    frame.area = area

    screen.st(['displacement', 'load', 'area','mean gap'],
              [disp0, load, area, mean_gap])

container.close()
