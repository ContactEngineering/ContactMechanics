#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   FromFile.py

@author Till Junge <till.junge@kit.edu>

@date   26 Jan 2015

@brief  Surface profile from file input

@section LICENCE

 Copyright (C) 2015 Till Junge

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

import os
import re
import xml.etree.ElementTree as ElementTree
from struct import unpack
from zipfile import ZipFile

import numpy as np

from .SurfaceDescription import NumpySurface


def read_matrix(fobj, size=None, factor=1.):
    """
    Reads a surface profile from a text file and presents in in a
    Surface-conformant manner. No additional parsing of meta-information is
    carried out.

    Keyword Arguments:
    fobj -- filename or file object
    """
    if not hasattr(fobj, 'read'):
        if not os.path.isfile(fobj):
            zfobj = fobj + ".gz"
            if os.path.isfile(zfobj):
                fobj = zfobj
            else:
                raise FileNotFoundError(
                    "No such file or directory: '{}(.gz)'".format(
                        fobj))
    return NumpySurface(factor*np.loadtxt(fobj), size=size)

NumpyTxtSurface = read_matrix  # pylint: disable=invalid-name


def read_asc(fobj, unit='m', x_factor=1.0, z_factor=1.0):
    # pylint: disable=too-many-branches,too-many-statements,invalid-name
    """
    Reads a surface profile from an generic asc file and presents it in a
    surface-conformant manner. Applies some heuristic to extract
    meta-information for different file formats. All units of the returned
    surface are in meters.

    Keyword Arguments:
    fobj -- filename or file object
    unit -- name of surface units, one of m, mm, μm/um, nm, A
    x_factor -- multiplication factor for size
    z_factor -- multiplication factor for height
    """
    _units = {'m': 1.0, 'mm': 1e-3, 'μm': 1e-6, 'um': 1e-6, 'nm': 1e-9,
              'A': 1e-10}

    if not hasattr(fobj, 'read'):
        if not os.path.isfile(fobj):
            raise FileNotFoundError(
                "No such file or directory: '{}(.gz)'".format(
                    fobj))
        fname = fobj
        fobj = open(fname)

    _float_regex = r'[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?'

    checks = list()
    # Resolution keywords
    checks.append((re.compile(r"\b(?:x-pixels|h)\b\s*=\s*([0-9]+)"), int,
                   "xres"))
    checks.append((re.compile(r"\b(?:y-pixels|w)\b\s*=\s*([0-9]+)"), int,
                   "yres"))

    # Size keywords
    checks.append((re.compile(r"\b(?:x-length)\b\s*=\s*("+_float_regex+")"),
                   float, "xsiz"))
    checks.append((re.compile(r"\b(?:y-length)\b\s*=\s*("+_float_regex+")"),
                   float, "ysiz"))

    # Unit keywords
    checks.append((re.compile(r"\b(?:x-unit)\b\s*=\s*(\w+)"), str, "xunit"))
    checks.append((re.compile(r"\b(?:y-unit)\b\s*=\s*(\w+)"), str, "yunit"))
    checks.append((re.compile(r"\b(?:z-unit)\b\s*=\s*(\w+)"), str, "zunit"))

    # Scale factor keywords
    checks.append((re.compile(r"(?:pixel\s+size)\s*=\s*("+_float_regex+")"),
                   float, "xfac"))
    checks.append((re.compile(
        (r"(?:height\s+conversion\s+factor\s+\(->\s+m\))\s*=\s*(" +
         _float_regex+")")),
                   float, "zfac"))

    xres = yres = xsiz = ysiz = xunit = yunit = zunit = xfac = yfac = None
    zfac = None

    def process_comment(line):
        "Find and interpret known comments in the header of the asc file"
        def check(line, reg, fun):
            "Check whether line fits a known comment syntax"
            match = reg.search(line)
            if match is not None:
                return fun(match.group(1))
            return None
        nonlocal xres, yres, xsiz, ysiz, xunit, yunit, zunit, data, xfac, yfac
        nonlocal zfac
        matches = {key: check(line, reg, fun)
                   for (reg, fun, key) in checks}
        if matches['xres'] is not None:
            xres = matches['xres']
        if matches['yres'] is not None:
            yres = matches['yres']
        if matches['xsiz'] is not None:
            xsiz = matches['xsiz']
        if matches['ysiz'] is not None:
            ysiz = matches['ysiz']
        if matches['xunit'] is not None:
            xunit = matches['xunit']
        if matches['yunit'] is not None:
            yunit = matches['yunit']
        if matches['zunit'] is not None:
            zunit = matches['zunit']
        if matches['xfac'] is not None:
            xfac = matches['xfac']
        if matches['zfac'] is not None:
            zfac = matches['zfac']

    data = []
    with fobj as file_handle:
        for line in file_handle:
            line_elements = line.strip().split()
            if len(line) > 0:
                try:
                    dummy = float(line_elements[0])
                    data += [[float(strval) for strval in line_elements]]
                except ValueError:
                    process_comment(line)
    data = np.array(data)
    nx, ny = data.shape
    if xres is not None and xres != nx:
        raise Exception(
            "The number of rows (={}) read from the file '{}' does "
            "not match the resolution in the file's metadata (={})."
            .format(nx, fname, xres))
    if yres is not None and yres != ny:
        raise Exception("The number of columns (={}) read from the file '{}' "
                        "does not match the resolution in the file's metadata "
                        "(={}).".format(ny, fname, yres))

    # Handle scale factors
    if xfac is not None and yfac is None:
        yfac = xfac
    elif xfac is None and yfac is not None:
        xfac = yfac
    if xfac is not None:
        if xsiz is None:
            xsiz = xfac*nx
        else:
            xsiz *= xfac
    if yfac is not None:
        if ysiz is None:
            ysiz = yfac*ny
        else:
            ysiz *= yfac
    if zfac is not None:
        data *= zfac

    # Handle units -> convert to target unit
    if xunit is None and zunit is not None:
        xunit = zunit
    if yunit is None and zunit is not None:
        yunit = zunit

    if xunit is not None:
        xsiz *= _units[xunit]/_units[unit]
    if yunit is not None:
        ysiz *= _units[yunit]/_units[unit]
    if zunit is not None:
        data *= _units[zunit]/_units[unit]

    if xsiz is None or ysiz is None:
        return NumpySurface(z_factor*data)
    else:
        return NumpySurface(z_factor*data, size=(x_factor*xsiz, x_factor*ysiz))

NumpyAscSurface = read_asc  # pylint: disable=invalid-name


def read_xyz(fobj):
    """
    Load xyz-file
    TODO: LARS_DOC
    Keyword Arguments:
    fobj -- filename or file object
    """
    # pylint: disable=invalid-name
    x, y, z = np.loadtxt(fobj, unpack=True)  # pylint: disable=invalid-name

    # Sort x-values into bins. Assume points on surface are equally spaced.
    dx = x[1]-x[0]
    binx = np.array(x/dx+0.5, dtype=int)
    n = np.bincount(binx)
    ny = n[0]
    assert np.all(n == ny)

    # Sort y-values into bins.
    dy = y[binx == 0][1]-y[binx == 0][0]
    biny = np.array(y/dy+0.5, dtype=int)
    n = np.bincount(biny)
    nx = n[0]
    assert np.all(n == nx)

    # Sort data into bins.
    data = np.zeros((nx, ny))
    data[binx, biny] = z

    # Sanity check. Should be covered by above asserts.
    value_present = np.zeros((nx, ny), dtype=bool)
    value_present[binx, biny] = True
    assert np.all(value_present)

    return NumpySurface(data, size=(dx*nx, dy*ny))


def read_x3p(fobj):
    """
    Load x3p-file.
    See: http://opengps.eu

    FIXME: Descriptive error messages. Probably needs to be made more robust.

    Keyword Arguments:
    fobj -- filename or file object
    """

    # Data types of binary container
    # See: https://sourceforge.net/p/open-gps/mwiki/X3p/
    dtype_map = {'I': np.dtype('<u2'),
                 'L': np.dtype('<u4'),
                 'F': np.dtype('f4'),
                 'D': np.dtype('f8')}

    with ZipFile(fobj, 'r') as x3p:
        xmlroot = ElementTree.parse(x3p.open('main.xml')).getroot()
        record1 = xmlroot.find('Record1')
        record3 = xmlroot.find('Record3')

        assert record1 is not None
        assert record3 is not None

        # Parse record1

        feature_type = record1.find('FeatureType')
        assert feature_type.text == 'SUR'
        axes = record1.find('Axes')
        cx = axes.find('CX')
        cy = axes.find('CY')
        cz = axes.find('CZ')

        assert cx.find('AxisType').text == 'I'
        assert cy.find('AxisType').text == 'I'
        assert cz.find('AxisType').text == 'A'

        xinc = float(cx.find('Increment').text)
        yinc = float(cy.find('Increment').text)

        datatype = cz.find('DataType').text
        dtype = dtype_map[datatype]

        # Parse record3
        matrix_dimension = record3.find('MatrixDimension')
        nx = int(matrix_dimension.find('SizeX').text)
        ny = int(matrix_dimension.find('SizeY').text)
        nz = int(matrix_dimension.find('SizeZ').text)

        assert nz == 1

        data_link = record3.find('DataLink')
        binfn = data_link.find('PointDataLink').text

        rawdata = x3p.open(binfn).read(nx*ny*dtype.itemsize)
        data = np.frombuffer(rawdata, count=nx*ny*nz,
                             dtype=dtype).reshape(nx, ny)

    return NumpySurface(data, size=(xinc*nx, yinc*ny))


def read_opd(fobj):
    """
    Load Wyko Vision OPD file.

    FIXME: Descriptive error messages. Probably needs to be made more robust.

    Keyword Arguments:
    fobj -- filename or file object
    """

    BLOCK_SIZE = 24
    def read_block(fobj):
        blkname = fobj.read(16).split(b'\0', 1)[0].decode('latin-1')
        blktype, blklen, blkattr = unpack('<hlH', fobj.read(8))
        return blkname, blktype, blklen, blkattr

    if not hasattr(fobj, 'read'):
        fobj = open(fobj, 'rb')

    # Header
    tmp = fobj.read(2)

    # Read directory block
    dirname, dirtype, dirlen, dirattr = read_block(fobj)
    if dirname != 'Directory':
        raise IOError("Error reading directory block. "
                      "Header is '{}', expected 'Directory'".format(dirname))
    num_blocks = dirlen//BLOCK_SIZE
    assert num_blocks*BLOCK_SIZE == dirlen

    blocks = []
    for i in range(num_blocks-1):
        blocks += [read_block(fobj)]

    data = None
    nx = None
    ny = None
    pixel_size = 1.0
    aspect = 1.0
    for n, t, l, a in blocks:
        if l <= 0:
            continue
        if n == 'RAW DATA' or n == 'RAW_DATA' or n == 'OPD' or n == 'Raw':
            if data is not None:
                raise IOError('Multiple data blocks encountered.')

            nx, ny, elsize = unpack('<HHH', fobj.read(6))
            if elsize == 1:
                dtype = np.dtype('c')
            elif elsize == 2:
                dtype = np.dtype('<i2')
            elif elsize == 4:
                dtype = np.dtype('f4')
            else:
                raise IOError("Don't know how to handle element size {}."
                              .format(elsize))
            data = np.fromfile(fobj, dtype=dtype,
                               count=nx*ny).reshape(nx, ny)
        elif n == 'Wavelength':
            wavelength, = unpack('<f', fobj.read(4))
        elif n == 'Mult':
            mult, = unpack('<H', fobj.read(2))
        elif n == 'Aspect':
            aspect, = unpack('<f', fobj.read(4))
        elif n == 'Pixel_size':
            pixel_size, = unpack('<f', fobj.read(4))
        else:
            fobj.read(l)

    fobj.close()

    if data is None:
        raise IOError('No data block encountered.')

    return NumpySurface(data, size=(nx*pixel_size, ny*pixel_size*aspect))


def read(fobj, format=None):
    if format is None:
        format = 'asc'
        if not hasattr(fobj, 'read'):
            format = os.path.splitext(fobj)[-1][1:]

    readers = {'xyz': read_xyz,
               'x3p': read_x3p}
    if format not in readers:
        return read_asc(fobj)
    else:
        return readers[format](fobj)
