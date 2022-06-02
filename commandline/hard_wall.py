#!/usr/bin/env python3
#
# Copyright 2016, 2020 Lars Pastewka
#           2019-2020 Antoine Sanner
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

"""
Command line front-end for hard wall calculations
"""

import sys
from argparse import ArgumentParser, ArgumentTypeError

import numpy as np

import ContactMechanics
from ContactMechanics import FreeFFTElasticHalfSpace, PeriodicFFTElasticHalfSpace
from SurfaceTopography import PlasticTopography, open_topography
from ContactMechanics import make_system, make_plastic_system
from ContactMechanics.Tools.Logger import Logger, quiet, screen
from ContactMechanics.IO.NetCDF import NetCDFContainer

###

# Total number of area/load/displacements to use
nsteps = 20

# Text output
logger = screen
versionstr = 'ContactMechanics version: {}'.format(ContactMechanics.__version__)
logger.pr(versionstr)
commandline = ' '.join(sys.argv[:])

unit_to_meters = {
    'A': 1e-10, 'nm': 1e-9, 'µm': 1e-6, 'mm': 1e-3, 'm': 1.0,
    'unknown': 1.0
    }


###

def next_step(system, surface, history=None, pentol=None, maxiter=None,
              logger=quiet):
    """
    Run a full contact calculation. Try to guess displacement such that areas
    are equally spaced on a log scale.

    Parameters
    ----------
    system : PyCo.System.SystemBase object
        The contact mechanical system.
    surface : SurfaceTopography.SurfaceTopography object
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
    profile = surface.heights()

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
        disp0 = -top + 0.01 * (top - middle)
    else:
        ref_area = np.log10(np.array(area + 1 / np.prod(surface.nb_grid_pts)))
        darea = np.append(ref_area[1:] - ref_area[:-1], -ref_area[-1])
        i = np.argmax(darea)
        if i == step - 1:
            disp0 = bot + 2 * (disp[-1] - bot)
        else:
            disp0 = (disp[i] + disp[i + 1]) / 2

    opt = system.minimize_proxy(offset=disp0, logger=logger, pentol=pentol,
                                maxiter=maxiter, verbose=arguments.verbose)
    c = opt.active_set
    f = opt.jac
    u = opt.x[:f.shape[0], :f.shape[1]]
    disp = np.append(disp, [disp0])
    gap = np.append(gap, [np.mean(u) - middle - disp0])
    current_load = f.sum() / np.prod(surface.physical_sizes)
    load = np.append(load, [current_load])
    current_area = (f > 0).sum() / np.prod(surface.nb_grid_pts)
    area = np.append(area, [current_area])
    converged = np.append(converged, np.array([opt.success], dtype=bool))
    logger.pr('disp = {}, area = {}, load = {}, converged = {}'
              .format(disp0, current_area, current_load, opt.success))

    # Sort by area
    disp, gap, load, area, converged = np.transpose(sorted(zip(disp, gap, load,
                                                               area, converged),
                                                           key=lambda x: x[3]))
    converged = np.array(converged, dtype=bool)

    return c, u, f, disp0, current_load, current_area, (disp, gap, load, area, converged)


def dump(txt, surface, u, f, offset=0):
    mean_elastic = np.mean(u)
    mean_rigid = np.mean(surface[...]) + offset
    load = f.sum()
    mean_pressure = load / np.prod(surface.physical_sizes)
    area = (f > 0).sum()
    fractional_area = area / np.prod(surface.nb_grid_pts)
    area *= surface.area_per_pt
    if substrate.young == 1:
        header = ['mean elastic ({})'.format(surface.info['unit']),
                  'mean rigid ({})'.format(surface.info['unit']),
                  'mean gap ({})'.format(surface.info['unit']),
                  'load (E* {}^2)'.format(surface.info['unit']),
                  'mean pressure (E*)',
                  'area ({}^2)'.format(surface.info['unit']),
                  'fractional area']
    else:
        header = ['mean elastic ({})'.format(surface.info['unit']),
                  'mean rigid ({})'.format(surface.info['unit']),
                  'mean gap ({})'.format(surface.info['unit']),
                  'load ([Units of E*] {}^2)'.format(surface.info['unit']),
                  'mean pressure ([Units of E*])',
                  'area ({}^2)'.format(surface.info['unit']),
                  'fractional area']
    data = [mean_elastic, mean_rigid, mean_elastic - mean_rigid, load,
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


def save_contact(fn, surface, substrate, pressure, macro=None):
    macrostr = ''
    if macro is not None:
        macrostr = '\n'.join(['{} = {}'.format(x, y) for x, y in macro])
    np.savetxt(fn, pressure, fmt='%i', header=versionstr + '\n' + commandline + '\n' + macrostr +
               'Contact map follows. Values are boolean.')


def save_pressure(fn, surface, substrate, pressure, macro=None):
    if substrate.young == 1:
        unitstr = 'Pressure values follow, they are reported in units of E*.'
    else:
        unitstr = 'This calculation was run with a contact modulus ' \
                  'E*={}.'.format(substrate.young)
    macrostr = ''
    if macro is not None:
        macrostr = '\n'.join(['{} = {}'.format(x, y) for x, y in macro])
    np.savetxt(fn, pressure, header=versionstr + '\n' + commandline + '\n' + macrostr + unitstr)


def save_gap(fn, surface, gap, macro=None):
    if surface.info['unit'] is None:
        unitstr = 'No unit information available.'
    else:
        unitstr = 'Gap values follow, they are reported in units of ' \
                  '{}.'.format(surface.info['unit'])
    macrostr = ''
    if macro is not None:
        macrostr = '\n'.join(['{} = {}'.format(x, y) for x, y in macro])
    np.savetxt(fn, gap, header=versionstr + '\n' + commandline + '\n' + macrostr + unitstr)


# Parse command line arguments

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
                    help="use contact/Young's modulus MODULUS",
                    metavar='MODULUS')
parser.add_argument('--poisson', dest='poisson', type=float, default=0.0,
                    help='use Poisson number POISSON',
                    metavar='POISSON')
parser.add_argument('--thickness', dest='thickness', type=float, default=None,
                    help='model a substrate of thickness THICKNESS on a rigid'
                         'substrate',
                    metavar='THICKNESS')
parser.add_argument('--hardness', dest='hardness', type=float,
                    help='use penetration hardness HARDNESS',
                    metavar='HARDNESS')
parser.add_argument('--maxiter', dest='maxiter', type=int, default=1000,
                    help='stop convergence after MAXITER iterations',
                    metavar='MAXITER')
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
parser.add_argument('--pressure-from-file', dest='pressure_from_fn', type=str,
                    help='compute contact area at external pressures given in '
                         'the file PRESSUREFN',
                    metavar='PRESSUREFN')
parser.add_argument('--physical_sizes', dest='physical_sizes', type=tuple2,
                    help='physical_sizes of surface is SIZE',
                    metavar='SIZE')
parser.add_argument('--physical_sizes-unit', dest='size_unit', type=str,
                    help='physical_sizes unit UNIT',
                    metavar='UNIT')
parser.add_argument('--height-fac', dest='height_fac', type=float,
                    help='scale all height by factor FAC',
                    metavar='FAC')
parser.add_argument('--height-unit', dest='height_unit', type=str,
                    help='height unit UNIT',
                    metavar='UNIT')
parser.add_argument('--pentol', dest='pentol', type=float,
                    help='tolerance for penetration of surface PENTOL',
                    metavar='PENTOL')
parser.add_argument('--contact-fn', dest='contact_fn', type=str,
                    help='filename for contact map CONTACTFN',
                    metavar='CONTACTFN')
parser.add_argument('--pressure-fn', dest='pressure_fn', type=str,
                    help='filename for pressure map PRESSUREFN',
                    metavar='PRESSUREFN')
parser.add_argument('--displ-fn', dest='displ_fn', type=str,
                    help='filename for displacement map DISPLFN',
                    metavar='DISPLFN')
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
logger.pr('poisson = {}'.format(arguments.poisson))
logger.pr('hardness = {}'.format(arguments.hardness))
logger.pr('thickness = {}'.format(arguments.thickness))
logger.pr('maxiter = {}'.format(arguments.maxiter))
logger.pr('displacement = {}'.format(arguments.displacement))
logger.pr('pressure = {}'.format(arguments.pressure))
logger.pr('pressure-from-file = {}'.format(arguments.pressure_from_fn))
logger.pr('physical_sizes = {}'.format(arguments.physical_sizes))
logger.pr('size_unit = {}'.format(arguments.size_unit))
logger.pr('height_fac = {}'.format(arguments.height_fac))
logger.pr('height_unit = {}'.format(arguments.height_unit))
logger.pr('pentol = {}'.format(arguments.pentol))
logger.pr('contact-fn = {}'.format(arguments.contact_fn))
logger.pr('pressure-fn = {}'.format(arguments.pressure_fn))
logger.pr('displ-fn = {}'.format(arguments.displ_fn))
logger.pr('gap-fn = {}'.format(arguments.gap_fn))
logger.pr('log-fn = {}'.format(arguments.log_fn))
logger.pr('netcdf-fn = {}'.format(arguments.netcdf_fn))

###

# Read a surface topography from a text file. Returns a SurfaceTopography.SurfaceTopography
# object.
reader = open_topography(arguments.filename)

# Set the *physical* physical_sizes of the surface. We here set it to equal the shape,
# i.e. the nb_grid_pts of the surface just open_topography. Size is returned by surface.physical_sizes
# and can be unknown, i.e. *None*.
unit = None
if arguments.size_unit is not None:  # overrides the unit
    unit = arguments.size_unit
    if reader.default_channel.unit is not None:
        logger.pr(f"overriding topography file unit {reader.default_channel.unit} with {unit}")
elif reader.default_channel.unit is None:
    unit = 'N/A'

surface = reader.topography(physical_sizes=arguments.physical_sizes, unit=unit)

if arguments.height_fac is not None or arguments.height_unit is not None:
    fac = 1.0
    if arguments.height_fac is not None:
        fac *= arguments.height_fac
    if arguments.height_unit is not None:
        fac *= unit_to_meters[arguments.height_unit] / unit_to_meters[surface.info['unit']]
    logger.pr('Rescaling surface heights by {}.'.format(fac))
    surface = surface.scale(fac)

logger.pr('SurfaceTopography has dimension of {} and physical_sizes of {} {}.'
          .format(surface.nb_grid_pts, surface.physical_sizes, surface.info['unit']))
logger.pr('RMS height = {}, RMS gradient = {}'.format(surface.rms_height_from_area(),
                                                      surface.rms_gradient()))
if arguments.detrend is not None:
    surface = surface.detrend(detrend_mode=arguments.detrend)
    logger.pr('After detrending: RMS height = {}, RMS gradient = {}'
              .format(surface.rms_height_from_area(), surface.rms_gradient()))

if arguments.hardness is not None:
    surface = PlasticTopography(surface, arguments.hardness)

# Initialize elastic half-space.
if arguments.boundary == 'periodic':
    substrate = PeriodicFFTElasticHalfSpace(surface.nb_grid_pts, arguments.modulus,
                                            surface.physical_sizes,
                                            thickness=arguments.thickness,
                                            poisson=arguments.poisson)
elif arguments.boundary == 'nonperiodic':
    if arguments.thickness is not None:
        raise ValueError('"thickness" arguments cannot be used with '
                         'nonperiodic boundaries.')
    substrate = FreeFFTElasticHalfSpace(
        surface.nb_grid_pts,
        arguments.modulus / (1 - arguments.poisson ** 2),
        surface.physical_sizes
        )
else:
    raise ValueError('Unknown boundary conditions: '
                     '{}'.format(arguments.boundary))

# Piece the full system together. In particular the PyCo.System.SystemBase
# object knows how to optimize the problem. For the hard wall interaction it
# will always use Polonsky & Keer's constrained conjugate gradient method.
if arguments.hardness is not None:
    system = make_plastic_system(substrate, surface)
else:
    system = make_system(substrate, surface)

###

# Create a NetCDF container to dump displacements and forces to.
container = None
if arguments.netcdf_fn is not None:
    container = NetCDFContainer(arguments.netcdf_fn, mode='w', double=True)
    container.set_shape(surface.nb_grid_pts)

if arguments.pressure is not None or arguments.pressure_from_fn is not None:
    if arguments.displacement is not None:
        raise ValueError('Please specify either displacement or pressure '
                         'range, not both.')

    # Run computation for a linear range of pressures
    if arguments.pressure is not None:
        pressure = arguments.pressure.split(',')
        if len(pressure) == 1:
            pressure = [float(pressure[0])]
        elif len(pressure) == 3:
            pressure = np.linspace(float(pressure[0]), float(pressure[1]), int(pressure[2]))
        else:
            print('Please specify either single pressure value or 3-tuple for '
                  'pressure range.')
            sys.exit(999)
    elif arguments.pressure_from_fn is not None:
        pressure = np.ravel(np.loadtxt(arguments.pressure_from_fn))

    # Additional log file for load and area
    txt = Logger(arguments.log_fn)

    for i, _pressure in enumerate(pressure):
        suffix = '.{}'.format(i)
        if len(pressure) == 1:
            suffix = ''
        external_force = _pressure * np.prod(surface.physical_sizes)
        logger.pr(f'external_force = {external_force}')
        opt = system.minimize_proxy(logger=logger,
                                    external_force=external_force,
                                    pentol=arguments.pentol,
                                    maxiter=arguments.maxiter,
                                    verbose=arguments.verbose)
        c = opt.active_set
        f = opt.jac
        u = opt.x[:f.shape[0], :f.shape[1]]
        logger.pr(f'displacement = {opt.offset}')
        logger.pr(f'pressure = {f.sum() / np.prod(surface.physical_sizes)} ({_pressure})')
        logger.pr(f'energy = {opt.fun}')
        logger.pr(f'fractional contact area = {(f > 0).sum() / np.prod(surface.nb_grid_pts)}')

        area = (f > 0).sum() / np.prod(surface.nb_grid_pts)
        load = _pressure
        disp0 = opt.offset
        dump_nc(container)
        macro = dump(txt, surface, u, f, opt.offset)

        if arguments.contact_fn is not None:
            save_contact(arguments.contact_fn + suffix, surface, substrate, c, macro=macro)
        if arguments.pressure_fn is not None:
            save_pressure(arguments.pressure_fn + suffix, surface, substrate, f / surface.area_per_pt, macro=macro)
        if arguments.displ_fn is not None:
            save_gap(arguments.displ_fn + suffix, surface, u, macro=macro)
        if arguments.gap_fn is not None:
            save_gap(arguments.gap_fn + suffix, surface, u - surface[...] - opt.offset, macro=macro)

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
        opt = system.minimize_proxy(offset=_displacement, logger=logger,
                                    pentol=arguments.pentol,
                                    maxiter=arguments.maxiter, kind='ref',
                                    verbose=arguments.verbose)
        c = opt.active_set
        f = opt.jac
        u = opt.x[:f.shape[0], :f.shape[1]]
        logger.pr('displacement = {} ({})'.format(opt.offset, _displacement))
        logger.pr('pressure = {}'.format(f.sum() / np.prod(surface.physical_sizes)))
        logger.pr('energy = {}'.format(opt.fun))
        logger.pr('fractional contact area = {}'
                  .format((f > 0).sum() / np.prod(surface.nb_grid_pts)))

        dump_nc(container)
        macro = dump(txt, surface, u, f, opt.offset)

        if arguments.contact_fn is not None:
            save_contact(arguments.contact_fn + suffix, surface, substrate, c, macro=macro)
        if arguments.pressure_fn is not None:
            save_pressure(arguments.pressure_fn + suffix, surface, substrate, f / surface.area_per_pt, macro=macro)
        if arguments.displ_fn is not None:
            save_gap(arguments.displ_fn + suffix, surface, u, macro=macro)
        if arguments.gap_fn is not None:
            save_gap(arguments.gap_fn + suffix, surface, u - surface[...] - opt.offset, macro=macro)

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

        c, u, f, disp0, load, area, history = \
            next_step(system, surface, history, pentol=arguments.pentol,
                      maxiter=arguments.maxiter, logger=logger)

        dump_nc(container)
        macro = dump(txt, surface, u, f, disp0)

        if arguments.contact_fn is not None:
            save_contact(arguments.contact_fn + suffix, surface, substrate, c, macro=macro)
        if arguments.pressure_fn is not None:
            save_pressure(arguments.pressure_fn + suffix, surface, substrate, f / surface.area_per_pt, macro=macro)
        if arguments.displ_fn is not None:
            save_gap(arguments.displ_fn + suffix, surface, u, macro=macro)
        if arguments.gap_fn is not None:
            save_gap(arguments.gap_fn + suffix, surface, u - surface[...] - disp0, macro=macro)

if container is not None:
    container.close()
