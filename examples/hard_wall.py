#! /usr/bin/env python3

"""
Automatically compute contact area, load and displacement for a rough surface.
Tries to guess displacements such that areas are equally spaced on a log scale.
"""

from argparse import ArgumentParser, ArgumentTypeError

import numpy as np
import PyCo
from PyCo.ContactMechanics import HardWall
from PyCo.SolidMechanics import PeriodicFFTElasticHalfSpace
from PyCo.Surface import read, DetrendedSurface
from PyCo.System import SystemFactory
from PyCo.Tools.Logger import Logger, quiet, screen
from PyCo.Tools.NetCDF import NetCDFContainer

###

# Total number of area/load/displacements to use
nsteps = 20

# Maximum number of iterations per data point
maxiter = 1000

# Text output
logger = screen
versionstr = 'PyCo version: {}'.format(PyCo.__version__)
logger.pr(versionstr)

###

def next_step(system, surface, history=None, logger=quiet):
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

        disp0 = -middle
    elif step == 1:
        disp0 = -top+0.01*(top-middle)
    else:
        ref_area = np.log10(np.array(area+1/np.prod(surface.shape)))
        darea = np.append(ref_area[1:]-ref_area[:-1], -ref_area[-1])
        i = np.argmax(darea)
        if i == step-1:
            disp0 = bot+2*(disp[-1]-bot)
        else:
            disp0 = (disp[i]+disp[i+1])/2

    opt = system.minimize_proxy(disp0, pentol=1e-3, maxiter=maxiter,
                                logger=logger)
    u = opt.x
    f = opt.jac
    disp = np.append(disp, [disp0])
    gap = np.append(gap, [np.mean(u)-middle-disp0])
    current_load = f.sum()/np.prod(surface.size)
    load = np.append(load, [current_load])
    current_area = (f>0).sum()/np.prod(surface.shape)
    area = np.append(area, [current_area])
    converged = np.append(converged, np.array([opt.success], dtype=bool))
    logger.pr('disp = {}, area = {}, load = {}, converged = {}' \
        .format(disp0, current_area, current_load, opt.success))

    # Sort by area
    disp, gap, load, area, converged = np.transpose(sorted(zip(disp, gap, load,
                                                               area, converged),
                                                    key=lambda x: x[3]))
    converged = np.array(converged, dtype=bool)

    return u, f, disp0, current_load, current_area, \
        (disp, gap, load, area, converged)

### Parse command line arguments

def tuple2(s):
    try:
        x, y = map(int, s.split(','))
        return x, y
    except:
        raise ArgumentTypeError('Size must be sx,sy')

parser = ArgumentParser(description='Run a contact mechanics calculation with'
                                    'a hard-wall interaction using Polonsky & '
                                    'Keers constrained conjugate gradient '
                                    'solver.')
parser.add_argument('filename', metavar='FILENAME', help='name of topography file')
parser.add_argument('-E', '--modulus', dest='modulus', type=float, default=1.0,
                    help='use contact modulus MODULUS',
                    metavar='MODULUS')
parser.add_argument('-p', '--pressure', dest='pressure', type=float,
                    help='compute contact area at external pressure PRESSURE',
                    metavar='PRESSURE')
parser.add_argument('-s', '--size', dest='size', type=tuple2,
                    help='compute contact area at external pressure PRESSURE',
                    metavar='SIZE')
parser.add_argument('-t', '--pentol', dest='pentol', type=float,
                    help='tolerance for penetration of surface PENTOL',
                    metavar='PENTOL')
parser.add_argument('-P', '--pressure-fn', dest='pressure_fn', type=str,
                    help='filename for pressure map PRESSUREFN',
                    metavar='PRESSUREFN')
parser.add_argument('-G', '--gap-fn', dest='gap_fn', type=str,
                    help='filename for gap map GAPFN',
                    metavar='GAPFN')
parser.add_argument('-L', '--log-fn', dest='log_fn', type=str,
                    default='hard_wall.out',
                    help='filename for log file LOGFN that contains final '
                         'area and load',
                    metavar='LOGFN')
parser.add_argument('-N', '--netcdf-fn', dest='netcdf_fn', type=str,
                    default='hard_wall.nc',
                    help='filename for NetCDF file NETCDFFN',
                    metavar='NETCDFFN')
arguments = parser.parse_args()
logger.pr('filename = {}'.format(arguments.filename))
logger.pr('modulus = {}'.format(arguments.modulus))
logger.pr('pressure = {}'.format(arguments.pressure))
logger.pr('size = {}'.format(arguments.size))
logger.pr('pentol = {}'.format(arguments.pentol))

###

# Read a surface topography from a text file. Returns a PyCo.Surface.Surface
# object.
surface = read(arguments.filename)
# Set the *physical* size of the surface. We here set it to equal the shape,
# i.e. the resolution of the surface just read. Size is returned by surface.size
# and can be unknown, i.e. *None*.
if arguments.size is not None:
    surface.size = arguments.size
if surface.size is None:
    surface.size = surface.shape

logger.pr('Surface has dimension of {} and size of {} {}.'.format(surface.shape,
                                                                  surface.size,
                                                                  surface.unit))
logger.pr('RMS height = {}, RMS slope = {}'.format(surface.compute_rms_height(),
                                                   surface.compute_rms_slope()))

# Initialize elastic half-space.
substrate = PeriodicFFTElasticHalfSpace(surface.shape, arguments.modulus,
                                        surface.size)
# Hard-wall interaction. This is a dummy object.
interaction = HardWall()
# Piece the full system together. In particular the PyCo.System.SystemBase
# object knows how to optimize the problem. For the hard wall interaction it
# will always use Polonsky & Keer's constrained conjugate gradient method.
system = SystemFactory(substrate, interaction, surface)

###

if arguments.pressure is not None:
    opt = system.minimize_proxy(
        external_force=arguments.pressure*np.prod(surface.size),
        pentol=arguments.pentol, maxiter=maxiter, logger=logger, kind='ref')
    u = opt.x
    f = opt.jac
    logger.pr('displacement = {}'.format(opt.offset))
    logger.pr('pressure = {}'.format(f.sum()/np.prod(surface.size)))
    logger.pr('fractional contact area = {}' \
        .format((f>0).sum()/np.prod(surface.shape)))
    if arguments.pressure_fn is not None:
        np.savetxt(arguments.pressure_fn, f/surface.area_per_pt,
                   header=versionstr)
    if arguments.gap_fn is not None:
        np.savetxt(arguments.gap_fn, u-surface[...]-opt.offset,
                   header=versionstr)
else:
    # Create a NetCDF container to dump displacements and forces to.
    container = NetCDFContainer(arguments.netcdf_fn, mode='w', double=True)
    container.set_shape(surface.shape)

    # Additional log file for load and area
    txt = Logger(arguments.log_fn)

    history = None
    for i in range(nsteps):
        displacements, forces, disp0, load, area, history = \
            next_step(system, surface, history, logger=logger)
        frame = container.get_next_frame()
        frame.displacements = displacements
        frame.forces = forces
        frame.displacement = disp0
        frame.load = load
        frame.area = area

        txt.st(['gap', 'load', 'area'],
               [np.mean(displacements)-disp0, load, area])

    container.close()