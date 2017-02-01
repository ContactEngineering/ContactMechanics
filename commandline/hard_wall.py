#! /usr/bin/env python3
"""
@file   hard_wall.py

@author Lars Pastewka <lars.pastewka@kit.edu>

@date   06 Oct 2016

@brief  Command line front-end for hard wall calculations

@section LICENCE

 Copyright (C) 2016 Lars Pastewka

PyCo is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation, either version 3, or (at
your option) any later version.

PyCo is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with GNU Emacs; see the file COPYING. If not, write to the
Free Software Foundation, Inc., 59 Temple Place - Suite 330,
Boston, MA 02111-1307, USA.
"""

import sys
from argparse import ArgumentParser, ArgumentTypeError

import numpy as np
import PyCo
from PyCo.ContactMechanics import HardWall
from PyCo.SolidMechanics import (FreeFFTElasticHalfSpace,
                                 PeriodicFFTElasticHalfSpace)
from PyCo.Surface import read, DetrendedSurface, ScaledSurface
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
commandline = ' '.join(sys.argv[:])

unit_to_meters = {'A': 1e-10, 'nm': 1e-9, 'µm': 1e-6, 'mm': 1e-3, 'm': 1.0,
                  'unknown': 1.0}

###

def next_step(system, surface, history=None, pentol=None, logger=quiet):
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

    opt = system.minimize_proxy(
        hardness=hardness,
        offset=disp0, pentol=pentol, maxiter=maxiter,
        logger=logger, kind='ref',
        verbose=arguments.verbose)
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

def dump(txt, surface, u, f, offset=0):
    mean_elastic = np.mean(u)
    mean_rigid = np.mean(surface[...])+offset
    load = f.sum()
    mean_pressure = load/np.prod(surface.size)
    area = (f>0).sum()
    fractional_area = area/np.prod(surface.shape)
    area *= surface.area_per_pt
    header = ['mean elastic ({})'.format(surface.unit),
              'mean rigid ({})'.format(surface.unit),
              'mean gap ({})'.format(surface.unit),
              'load (E* {}^2)'.format(surface.unit),
              'mean pressure (E*)',
              'area ({}^2)'.format(surface.unit),
              'fractional area']
    data = [mean_elastic, mean_rigid, mean_elastic-mean_rigid, load,
            mean_pressure, area, fractional_area]
    txt.st(header, data)
    return zip(header, data)

def dump_nc(container):
    if container is not None:
        frame = container.get_next_frame()
        frame.displacements = u
        frame.forces = f
        frame.displacement = disp0
        frame.load = load
        frame.area = area

def save_pressure(fn, surface, substrate, pressure, macro=None):
    if substrate.young == 1:
        unitstr = 'Pressure values follow, they are reported in units of E*.'
    else:
        unitstr = 'This calculation was run with a contact modulus ' \
                  'E*={}.'.format(substrate.young)
    macrostr = ''
    if macro is not None:
        macrostr = '\n'.join(['{} = {}'.format(x, y) for x, y in macro])
    np.savetxt(fn, pressure, header=versionstr+'\n'+commandline+'\n'+macrostr+
               unitstr)

def save_gap(fn, surface, gap, macro=None):
    if surface.unit is None:
        unitstr = 'No unit information available.'
    else:
        unitstr = 'Gap values follow, they are repoted in units of ' \
                  '{}.'.format(surface.unit)
    macrostr = ''
    if macro is not None:
        macrostr = '\n'.join(['{} = {}'.format(x, y) for x, y in macro])
    np.savetxt(fn, gap, header=versionstr+'\n'+commandline+'\n'+macrostr+
                               unitstr)

### Parse command line arguments

def tuple2(s):
    try:
        x, y = (float(x) for x in s.split(','))
        return x, y
    except:
        raise ArgumentTypeError('Size must be sx,sy')

parser = ArgumentParser(description='Run a contact mechanics calculation with'
                                    'a hard-wall interaction using Polonsky & '
                                    'Keers constrained conjugate gradient '
                                    'solver.')
parser.add_argument('filename', metavar='FILENAME', help='name of topography file')
parser.add_argument('--detrend', dest='detrend', type=str,
                    help='detrend surface before contact calculation, DETREND '
                         'can be one of "height", "slope" or "curvature"',
                    metavar='DETREND')
parser.add_argument('--boundary', dest='boundary', type=str,
                    default='periodic',
                    help='specify boundary conditions; '
                         'BOUNDARY=periodic|nonperiodic',
                    metavar='BOUNDARY')
parser.add_argument('--modulus', dest='modulus', type=float, default=1.0,
                    help='use contact modulus MODULUS',
                    metavar='MODULUS')
parser.add_argument('--hardness', dest='hardness', type=float,
                    help='use penetration hardness HARDNESS',
                    metavar='HARDNESS')
parser.add_argument('--displacement', dest='displacement', type=str,
                    help='compute contact area at displacement DISPLACEMENT; '
                         'specify displacement range by using a 3-tuple '
                         'DISPLACEMENT=min,max,steps',
                    metavar='DISPLACEMENT')
parser.add_argument('--pressure', dest='pressure', type=str,
                    help='compute contact area at external pressure PRESSURE; '
                         'specify pressure range by using a 3-tuple '
                         'PRESSURE=min,max,steps',
                    metavar='PRESSURE')
parser.add_argument('--size', dest='size', type=tuple2,
                    help='size of surface is SIZE',
                    metavar='SIZE')
parser.add_argument('--size-unit', dest='size_unit', type=str,
                    help='size unit UNIT',
                    metavar='UNIT')
parser.add_argument('--height-unit', dest='height_unit', type=str,
                    help='height unit UNIT',
                    metavar='UNIT')
parser.add_argument('--pentol', dest='pentol', type=float,
                    help='tolerance for penetration of surface PENTOL',
                    metavar='PENTOL')
parser.add_argument('--pressure-fn', dest='pressure_fn', type=str,
                    help='filename for pressure map PRESSUREFN',
                    metavar='PRESSUREFN')
parser.add_argument('--gap-fn', dest='gap_fn', type=str,
                    help='filename for gap map GAPFN',
                    metavar='GAPFN')
parser.add_argument('--log-fn', dest='log_fn', type=str,
                    default='hard_wall.out',
                    help='filename for log file LOGFN that contains final '
                         'area and load',
                    metavar='LOGFN')
parser.add_argument('--netcdf-fn', dest='netcdf_fn', type=str,
                    default=None,
                    help='filename for NetCDF file NETCDFFN',
                    metavar='NETCDFFN')
parser.add_argument('--verbose', dest='verbose', action='store_true',
                    default=False,
                    help='enable verbose output')
arguments = parser.parse_args()
logger.pr('filename = {}'.format(arguments.filename))
logger.pr('detrend = {}'.format(arguments.detrend))
logger.pr('boundary = {}'.format(arguments.boundary))
logger.pr('modulus = {}'.format(arguments.modulus))
logger.pr('hardness = {}'.format(arguments.hardness))
logger.pr('displacement = {}'.format(arguments.displacement))
logger.pr('pressure = {}'.format(arguments.pressure))
logger.pr('size = {}'.format(arguments.size))
logger.pr('size_unit = {}'.format(arguments.size_unit))
logger.pr('height_unit = {}'.format(arguments.height_unit))
logger.pr('pentol = {}'.format(arguments.pentol))
logger.pr('pressure-fn = {}'.format(arguments.pressure_fn))
logger.pr('gap-fn = {}'.format(arguments.gap_fn))
logger.pr('log-fn = {}'.format(arguments.log_fn))
logger.pr('netcdf-fn = {}'.format(arguments.netcdf_fn))

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
if arguments.size_unit is not None:
    surface.unit = arguments.size_unit
if arguments.height_unit is not None:
    surface = ScaledSurface(surface, unit_to_meters[arguments.height_unit]/unit_to_meters[surface.unit])

logger.pr('Surface has dimension of {} and size of {} {}.'.format(surface.shape,
                                                                  surface.size,
                                                                  surface.unit))
logger.pr('RMS height = {}, RMS slope = {}'.format(surface.compute_rms_height(),
                                                   surface.compute_rms_slope()))
if arguments.detrend is not None:
    surface = DetrendedSurface(surface, detrend_mode=arguments.detrend)
    logger.pr('After detrending: RMS height = {}, RMS slope = {}' \
        .format(surface.compute_rms_height(), surface.compute_rms_slope()))

if arguments.hardness is not None:
    hardness = arguments.hardness*surface.area_per_pt
else:
    hardness = None

# Initialize elastic half-space.
if arguments.boundary == 'periodic':
    substrate = PeriodicFFTElasticHalfSpace(surface.shape, arguments.modulus,
                                            surface.size)
elif arguments.boundary == 'nonperiodic':
    substrate = FreeFFTElasticHalfSpace(surface.shape, arguments.modulus,
                                        surface.size)
else:
    raise ValueError('Unknown boundary conditions: '
                     '{}'.format(arguments.boundary))

# Hard-wall interaction. This is a dummy object.
interaction = HardWall()
# Piece the full system together. In particular the PyCo.System.SystemBase
# object knows how to optimize the problem. For the hard wall interaction it
# will always use Polonsky & Keer's constrained conjugate gradient method.
system = SystemFactory(substrate, interaction, surface)

###

# Create a NetCDF container to dump displacements and forces to.
container = None
if arguments.netcdf_fn is not None:
    container = NetCDFContainer(arguments.netcdf_fn, mode='w', double=True)
    container.set_shape(surface.shape)

if arguments.pressure is not None:
    if arguments.displacement is not None:
        raise ValueError('Please specify either displacement or pressure '
                         'range, not both.')

    # Run computation for a linear range of pressures
    pressure = arguments.pressure.split(',')
    if len(pressure) == 1:
        pressure = [float(pressure[0])]
    elif len(pressure) == 3:
        pressure = np.linspace(*[float(x) for x in pressure])
    else:
        print('Please specify either single pressure value or 3-tuple for '
              'pressure range.')
        sys.exit(999)

    # Additional log file for load and area
    txt = Logger(arguments.log_fn)

    for i, _pressure in enumerate(pressure):
        suffix = '.{}'.format(i)
        if len(pressure) == 1:
            suffix = ''
        opt = system.minimize_proxy(
            hardness=hardness,
            external_force=_pressure*np.prod(surface.size),
            pentol=arguments.pentol, maxiter=maxiter, logger=logger, kind='ref',
            verbose=arguments.verbose)
        u = opt.x
        f = opt.jac
        logger.pr('displacement = {}'.format(opt.offset))
        logger.pr('pressure = {} ({})'.format(f.sum()/np.prod(surface.size),
                                              _pressure))
        logger.pr('energy = {}'.format(opt.fun))
        logger.pr('fractional contact area = {}' \
            .format((f>0).sum()/np.prod(surface.shape)))

        dump_nc(container)
        macro = dump(txt, surface, u, f, opt.offset)

        if arguments.pressure_fn is not None:
            save_pressure(arguments.pressure_fn+suffix, surface, substrate,
                          f/surface.area_per_pt, macro=macro)
        if arguments.gap_fn is not None:
            save_gap(arguments.gap_fn+suffix, surface,
                     u-surface[...]-opt.offset, macro=macro)

elif arguments.displacement is not None:
    # Run computation for a linear range of displacements

    displacement = arguments.displacement.split(',')
    if len(displacement) == 1:
        displacement = [float(displacement[0])]
    elif len(displacement) == 3:
        displacement = np.linspace(*[float(x) for x in displacement])
    else:
        print('Please specify either single displacement value or 3-tuple for '
              'displacement range.')
        sys.exit(999)

    # Additional log file for load and area
    txt = Logger(arguments.log_fn)

    for i, _displacement in enumerate(displacement):
        suffix = '.{}'.format(i)
        if len(displacement) == 1:
            suffix = ''
        opt = system.minimize_proxy(
            hardness=hardness,
            offset=_displacement, pentol=arguments.pentol, maxiter=maxiter,
            logger=logger, kind='ref',
            verbose=arguments.verbose)
        u = opt.x
        f = opt.jac
        logger.pr('displacement = {} ({})'.format(opt.offset, _displacement))
        logger.pr('pressure = {}'.format(f.sum()/np.prod(surface.size)))
        logger.pr('energy = {}'.format(opt.fun))
        logger.pr('fractional contact area = {}' \
            .format((f>0).sum()/np.prod(surface.shape)))

        dump_nc(container)
        macro = dump(txt, surface, u, f, opt.offset)

        if arguments.pressure_fn is not None:
            save_pressure(arguments.pressure_fn+suffix, surface, substrate,
                          f/surface.area_per_pt, macro=macro)
        if arguments.gap_fn is not None:
            save_gap(arguments.gap_fn+suffix, surface,
                     u-surface[...]-opt.offset, macro=macro)

else:
    # Run computation automatically such that area is equally spaced on
    # a log scale. This is the default when no other command line options are
    # given.

    # Additional log file for load and area
    txt = Logger(arguments.log_fn)

    history = None
    for i in range(nsteps):
        suffix = '.{}'.format(i)
        if nsteps == 1:
            suffix = ''

        u, f, disp0, load, area, history = \
            next_step(system, surface, history, pentol=arguments.pentol,
                      logger=logger)

        dump_nc(container)
        macro = dump(txt, surface, u, f, disp0)

        if arguments.pressure_fn is not None:
            save_pressure(arguments.pressure_fn+suffix, surface, substrate,
                          f/surface.area_per_pt, macro=macro)
        if arguments.gap_fn is not None:
            save_gap(arguments.gap_fn+suffix, surface,  u-surface[...]-disp0,
                     macro=macro)

if container is not None:
    container.close()
