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
from ContactMechanics.PipelineFunction import _contact_steps
from ContactMechanics.Tools.Logger import Logger, screen
from ContactMechanics.IO.NetCDF import NetCDFContainer

###

# Default number of automatically chosen contact steps
NSTEPS = 20

unit_to_meters = {
    'A': 1e-10, 'nm': 1e-9, 'µm': 1e-6, 'mm': 1e-3, 'm': 1.0,
    'unknown': 1.0
}


###

def dump(txt, surface, substrate, u, f, offset=0, converged=True):
    mean_elastic = np.mean(u)
    mean_rigid = np.mean(surface[...]) + offset
    load = f.sum()
    mean_pressure = load / np.prod(surface.physical_sizes)
    area = (f > 0).sum()
    fractional_area = area / np.prod(surface.nb_grid_pts)
    area *= surface.area_per_pt
    if substrate.young == 1:
        header = ['mean elastic ({})'.format(surface.unit),
                  'mean rigid ({})'.format(surface.unit),
                  'mean gap ({})'.format(surface.unit),
                  'load (E* {}^2)'.format(surface.unit),
                  'mean pressure (E*)',
                  'area ({}^2)'.format(surface.unit),
                  'fractional area',
                  'converged']
    else:
        header = ['mean elastic ({})'.format(surface.unit),
                  'mean rigid ({})'.format(surface.unit),
                  'mean gap ({})'.format(surface.unit),
                  'load ([Units of E*] {}^2)'.format(surface.unit),
                  'mean pressure ([Units of E*])',
                  'area ({}^2)'.format(surface.unit),
                  'fractional area',
                  'converged']
    data = [mean_elastic, mean_rigid, mean_elastic - mean_rigid, load,
            mean_pressure, area, fractional_area, int(converged)]
    txt.st(header, data)
    return zip(header, data)


def dump_nc(container, u, f, disp0, load, area):
    if container is not None:
        frame = container.get_next_frame()
        frame.displacements = u
        frame.forces = f
        frame.displacement = disp0
        frame.load = load
        frame.area = area


def _macro_header(header, macro):
    macrostr = ''
    if macro is not None:
        macrostr = '\n'.join(['{} = {}'.format(x, y) for x, y in macro]) + '\n'
    return header + '\n' + macrostr


def save_contact(fn, header, contact, macro=None):
    np.savetxt(
        fn,
        contact,
        fmt='%i',
        header=_macro_header(header, macro) + 'Contact map follows. Values are boolean.')


def save_pressure(fn, header, substrate, pressure, macro=None):
    if substrate.young == 1:
        unitstr = 'Pressure values follow, they are reported in units of E*.'
    else:
        unitstr = 'This calculation was run with a contact modulus ' \
                  'E*={}.'.format(substrate.young)
    np.savetxt(fn, pressure, header=_macro_header(header, macro) + unitstr)


def save_gap(fn, header, surface, gap, macro=None):
    if surface.unit is None:
        unitstr = 'No unit information available.'
    else:
        unitstr = 'Gap values follow, they are reported in units of ' \
                  '{}.'.format(surface.unit)
    np.savetxt(fn, gap, header=_macro_header(header, macro) + unitstr)


# Parse command line arguments

def tuple2(s):
    try:
        x, y = (float(x) for x in s.split(','))
        return x, y
    except ValueError:
        raise ArgumentTypeError('Size must be sx,sy')
    except TypeError:
        raise ArgumentTypeError('Size must be sx,sy')


def _parse_range(s, name):
    """Parse a single value or a range min,max,steps"""
    values = s.split(',')
    if len(values) == 1:
        return [float(values[0])]
    elif len(values) == 3:
        return np.linspace(float(values[0]), float(values[1]), int(values[2]))
    raise ValueError('Please specify either single {0} value or 3-tuple for {0} range.'.format(name))


def main(args=None):
    parser = ArgumentParser(description='Run a contact mechanics calculation with '
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
    parser.add_argument('--maxiter', dest='maxiter', type=int, default=None,
                        help='stop convergence after MAXITER iterations '
                             '(default: default of the solver)',
                        metavar='MAXITER')
    parser.add_argument('--nsteps', dest='nsteps', type=int, default=NSTEPS,
                        help='number of automatically chosen contact steps if '
                             'neither displacement nor pressure is given '
                             '(default: {})'.format(NSTEPS),
                        metavar='NSTEPS')
    parser.add_argument('--displacement', dest='displacement', type=str,
                        help='compute contact area at displacement DISPLACEMENT; '
                             'specify displacement range by using a 3-tuple '
                             'DISPLACEMENT=min,max,steps (use '
                             '--displacement=DISPLACEMENT if DISPLACEMENT '
                             'starts with a minus sign)',
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
    arguments = parser.parse_args(args)

    # Text output
    logger = screen
    versionstr = 'ContactMechanics version: {}'.format(ContactMechanics.__version__)
    logger.pr(versionstr)
    commandline = ' '.join(sys.argv[:1] + (sys.argv[1:] if args is None else list(args)))
    header = versionstr + '\n' + commandline

    for name, value in sorted(vars(arguments).items()):
        logger.pr('{} = {}'.format(name, value))

    nb_modes = sum(x is not None for x in (arguments.displacement, arguments.pressure, arguments.pressure_from_fn))
    if nb_modes > 1:
        raise ValueError('Please specify only one of displacement, pressure or pressure-from-file.')

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
            if surface.unit not in unit_to_meters:
                raise ValueError('Cannot convert heights from unit {} to {}. Please specify the unit of the '
                                 'topography with --physical_sizes-unit.'.format(surface.unit, arguments.height_unit))
            fac *= unit_to_meters[arguments.height_unit] / unit_to_meters[surface.unit]
        logger.pr('Rescaling surface heights by {}.'.format(fac))
        surface = surface.scale(fac)

    logger.pr('SurfaceTopography has dimension of {} and physical_sizes of {} {}.'
              .format(surface.nb_grid_pts, surface.physical_sizes, surface.unit))
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
        system = make_plastic_system(substrate=substrate, surface=surface)
    else:
        system = make_system(substrate=substrate, surface=surface)

    ###

    # Create a NetCDF container to dump displacements and forces to.
    container = None
    if arguments.netcdf_fn is not None:
        container = NetCDFContainer(arguments.netcdf_fn, mode='w', double=True)
        container.set_shape(surface.nb_grid_pts)

    if arguments.displacement is not None:
        # Run computation for a linear range of displacements
        steps = dict(offsets=_parse_range(arguments.displacement, 'displacement'))
    elif arguments.pressure is not None:
        # Run computation for a linear range of pressures
        steps = dict(pressures=_parse_range(arguments.pressure, 'pressure'))
    elif arguments.pressure_from_fn is not None:
        steps = dict(pressures=np.ravel(np.loadtxt(arguments.pressure_from_fn)))
    else:
        # Run computation automatically such that area is equally spaced on
        # a log scale. This is the default when no other command line options are
        # given.
        steps = dict(nsteps=arguments.nsteps)
    nb_steps = len(next(iter(steps.values()))) if 'nsteps' not in steps else steps['nsteps']

    # Additional log file for load and area
    txt = Logger(arguments.log_fn)

    optimizer_kwargs = dict(logger=logger, verbose=arguments.verbose)
    for i, (u, gap, pressure, contact, offset, mean_pressure, fractional_area, history) in enumerate(
            _contact_steps(system, pentol=arguments.pentol, maxiter=arguments.maxiter,
                           optimizer_kwargs=optimizer_kwargs, **steps)):
        suffix = '' if nb_steps == 1 else '.{}'.format(i)
        converged = history[-1][-1]
        f = pressure * surface.area_per_pt

        logger.pr('displacement = {}'.format(offset))
        logger.pr('pressure = {}'.format(mean_pressure))
        logger.pr('energy = {}'.format(np.sum(f * u) / 2))
        logger.pr('fractional contact area = {}'.format(fractional_area))
        logger.pr('converged = {}'.format(converged))

        dump_nc(container, u, f, offset, mean_pressure, fractional_area)
        macro = dump(txt, surface, substrate, u, f, offset, converged)

        if arguments.contact_fn is not None:
            save_contact(arguments.contact_fn + suffix, header, contact, macro=macro)
        if arguments.pressure_fn is not None:
            save_pressure(arguments.pressure_fn + suffix, header, substrate, pressure, macro=macro)
        if arguments.displ_fn is not None:
            save_gap(arguments.displ_fn + suffix, header, surface, u, macro=macro)
        if arguments.gap_fn is not None:
            save_gap(arguments.gap_fn + suffix, header, surface, gap, macro=macro)

    if container is not None:
        container.close()
