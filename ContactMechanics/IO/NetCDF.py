#
# Copyright 2016, 2019-2020 Lars Pastewka
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
Output surface topography and contact deformation to a structured NetCDF
database.
"""

from __future__ import print_function

import numbers
from math import sqrt

import numpy as np

try:
    from netCDF4 import Dataset

    __have_netcdf4__ = True
except ImportError:
    from pupynere import NetCDFFile

    __have_netcdf4__ = False


class NetCDFContainerFrame(object):
    def __init__(self, parent, i):
        self._parent = parent
        self._i = i
        if self._i < 0:
            self._i += len(parent)

    def _create_if_missing(self, name, value, shape=None):
        if name not in self._parent._data.variables:
            if isinstance(value, numbers.Integral):
                self._parent._data.createVariable(name, 'i4', ('frame',))
            elif isinstance(value, numbers.Real):
                self._parent._data.createVariable(name, 'f8', ('frame',))
            elif isinstance(value, np.ndarray):
                if shape is None:
                    shape = value.shape
                if shape == ():
                    self._parent._data.createVariable(name, 'f8', ('frame',))
                elif len(shape) == len(self._parent._shape) and np.all(
                        np.array(shape) == np.array(self._parent._shape)):
                    if len(self._parent._shape) == 3:
                        self._parent._data.createVariable(
                            name,
                            self._parent._float_str,
                            ('frame', 'ndof', 'nx', 'ny',)
                        )
                    else:
                        self._parent._data.createVariable(
                            name,
                            self._parent._float_str,
                            ('frame', 'nx', 'ny',)
                        )
                elif len(shape) == 2 and np.all(
                        np.array(shape) == np.array(self._parent._shape2)):
                    self._parent._data.createVariable(name,
                                                      self._parent._float_str,
                                                      ('frame', 'nx', 'ny',))
                else:
                    raise RuntimeError('Not sure how to guess NetCDF type for '
                                       'field "{0}" which is a numpy ndarray '
                                       'with type {1} and shape {2}.'
                                       .format(name, value.dtype, value.shape))
            else:
                raise RuntimeError('Not sure how to guess NetCDF type for '
                                   'field "{0}" with type {1}.'
                                   .format(name, type(value)))

    def _mangle_name(self, name):
        # Name mangling
        if self._parent._is_amber:
            if name == 'u':
                return 'coordinates'
            elif name == 'f':
                return 'forces'
        return name

    def __getattr__(self, name):
        if name[0] == '_':
            return self.__dict__[name]

        name = self._mangle_name(name)

        attr = self._parent._data.variables[name][self._i]
        # Reshape
        if self._parent._is_amber:
            nx, ny = self._parent._shape
            attr.shape = (nx, ny, 3)
        return attr

    def __setattr__(self, name, value):
        if name[0] == '_':
            object.__setattr__(self, name, value)
            return

        name = self._mangle_name(name)
        self._create_if_missing(name, value)

        self._parent._data.variables[name][self._i] = value

    def __getitem__(self, name):
        return self.__getattr__(name)

    def __setitem__(self, name, value):
        return self.__setattr__(name, value)

    def get_index(self):
        return self._i

    index = property(get_index)

    def get_traj(self):
        return self._parent

    traj = property(get_traj)

    def set_grid(self, name, value, x0=0, y0=0):
        """
        Assing a value to a certain grid, where the grid starts at x0,y0. Grid
        can be smaller than the dimensions in the file. Padding region is then
        set to undefined.
        """

        name = self._mangle_name(name)
        self._create_if_missing(name, value, shape=self._parent._shape)

        nx, ny = value.shape
        # print x0, y0, x0+nx, y0+ny
        self._parent._data.variables[name][self._i, x0:x0 + nx, y0:y0 + ny] = \
            value

    def sync(self):
        self._parent.sync()


class NetCDFContainer(object):
    def __init__(self, fn, frame=0, double=False, store_force=False,
                 mode='r', format='NETCDF4'):
        self._fn = fn
        self._data = None
        try:
            if __have_netcdf4__:
                self._data = Dataset(fn, mode, format=format)
            else:
                if mode == 'ws':
                    mode = 'w'
                self._data = NetCDFFile(fn, mode)
        except RuntimeError as e:
            raise RuntimeError('Error opening file "{0}": {1}'.format(fn, e))

        if double:
            self._float_str = 'f8'
        else:
            self._float_str = 'f4'

        self._store_force = store_force

        self._is_amber = False

        if mode[0] == 'w':
            self._data.program = 'PyCo'
            self._data.programVersion = 'N/A'
            self._is_defined = False
        else:
            if 'nx' in self._data.dimensions and 'ny' in self._data.dimensions:
                try:
                    self._shape = (len(self._data.dimensions['nx']),
                                   len(self._data.dimensions['ny']))
                except TypeError:
                    self._shape = (self._data.dimensions['nx'],
                                   self._data.dimensions['ny'])
                self._shape2 = self._shape
            elif 'atom' in self._data.dimensions:
                n = self._data.dimensions['atom']
                nx = int(sqrt(n))
                assert nx * nx == n
                self._shape = (nx, nx)
                self._shape2 = (nx, nx)

                self._is_amber = True
            else:
                raise RuntimeError('Unknown NetCDF convention used for file '
                                   '%s.' % fn)

            self._is_defined = True

        if frame < 0:
            self._cur_frame = len(self) + frame
        else:
            self._cur_frame = frame

    def __del__(self):
        if self._data is not None:
            self._data.close()

    def _define_file_structure(self, shape):
        # print 'defining file structure, shape = {0}'.format(shape)
        self._shape = shape

        if len(shape) == 3:
            ndof, nx, ny = shape
        else:
            ndof = 1
            nx, ny = shape

        self._shape2 = (nx, ny)

        if 'frame' not in self._data.dimensions:
            self._data.createDimension('frame', None)
        if ndof > 1 and 'ndof' not in self._data.dimensions:
            self._data.createDimension('ndof', ndof)
        if 'nx' not in self._data.dimensions:
            self._data.createDimension('nx', nx)
        if 'ny' not in self._data.dimensions:
            self._data.createDimension('ny', ny)

        self._data.sync()

        self._is_defined = True

    def set_shape(self, x, ndof=None):
        try:
            shape = x.shape
        except AttributeError:
            shape = x
        if not self._is_defined:
            if ndof is None or ndof == 1:
                self._define_file_structure(shape)
            else:
                self._define_file_structure([ndof] + list(shape))
        else:
            if ndof is None or ndof == 1:
                if not np.all(np.array(shape) == np.array(self._shape)):
                    raise RuntimeError('Shape mismatch: NetCDF file has shape '
                                       '{0} x {1}, but someone is trying to '
                                       'override this with shape {2} x {3}.'
                                       .format(self._shape[0], self._shape[1],
                                               shape[0], shape[1]))
            else:
                assert np.all(np.array([ndof] + list(shape)) ==
                              np.array(self._shape))

    def __len__(self):
        try:
            length = len(self._data.dimensions['frame'])
        except TypeError:
            length = self._data.dimensions['frame']
        return length

    def close(self):
        if self._data is not None:
            self._data.close()
            self._is_defined = False
            self._data = None

    def has_h(self):
        return 'h' in self._data.variables

    def set_rigid_surface(self, h, ndof=None):
        self.set_shape(h, ndof=ndof)
        nx, ny = self._shape2
        hnx, hny = h.shape

        if 'h' not in self._data.variables:
            if hnx != nx or hny != ny:
                if 'rigid_nx' not in self._data.dimensions:
                    self._data.createDimension('rigid_nx', hnx)
                if 'rigid_ny' not in self._data.dimensions:
                    self._data.createDimension('rigid_ny', hny)
                self._data.createVariable('h', 'f8', ('rigid_nx',
                                                      'rigid_ny',))
            else:
                self._data.createVariable('h', 'f8', ('nx', 'ny',))

        self._data.variables['h'][:, :] = h

    # Backward compatibility
    set_h = set_rigid_surface

    def set_elastic_surface(self, h):
        if 'elastic_surface' not in self._data.variables:
            self._data.createVariable('elastic_surface', 'f8',
                                      ('nx', 'ny',)
                                      )

        self._data.variables['elastic_surface'][:, :] = h

    def get_rigid_surface(self):
        return self._data.variables['h'][:, :]

    # Backward compatibility
    get_h = get_rigid_surface

    def get_elastic_surface(self):
        return self._data.variables['elastic_surface']

    def get_filename(self):
        return self._fn

    def get_next_frame(self):
        frame = NetCDFContainerFrame(self, self._cur_frame)
        self._cur_frame += 1
        return frame

    def set_cursor(self, cur_frame):
        self._cur_frame = cur_frame

    def get_cursor(self):
        return self._cur_frame

    def __getattr__(self, name):
        if name[0] == '_':
            return self.__dict__[name]

        if name in self._data.variables:
            return self._data.variables[name][...]

        return self._data.__getattr__(name)

    def __setattr__(self, name, value):
        if name[0] == '_':
            return object.__setattr__(self, name, value)

        if isinstance(value, np.ndarray) and value.shape != ():
            if name not in self._data.variables:
                if len(value.shape) == len(self._shape) and \
                        np.all(np.array(value.shape) == np.array(self._shape)):
                    self._data.createVariable(name, 'f8', ('nx', 'ny',))
                else:
                    raise RuntimeError('Not sure how to guess NetCDF type for '
                                       'field "{0}" which is a numpy ndarray '
                                       'with type {1} and shape {2}.'
                                       .format(name, value.dtype, value.shape))

            self._data.variables[name][:, :] = value
            return

        return self._data.__setattr__(name, value)

    def __setitem__(self, i, value):
        if isinstance(i, str):
            return self.__setattr__(i, value)
        raise RuntimeError('Cannot set full frame.')

    def __getitem__(self, i):
        if isinstance(i, str):
            return self.__getattr__(i)
        if isinstance(i, slice):
            return [NetCDFContainerFrame(self, j)
                    for j in range(*i.indices(len(self)))]
        return NetCDFContainerFrame(self, i)

    def __iter__(self):
        for i in range(len(self)):
            yield NetCDFContainerFrame(self, i)

    def get_size(self):
        return self._shape

    def sync(self):
        self._data.sync()

    def __str__(self):
        return self._fn


###

def open(fn, mode='r', frame=None, **kwargs):
    if isinstance(fn, NetCDFContainer):
        return fn
    i = fn.find('@')
    if i > 0:
        n = int(fn[i + 1:])
        fn = fn[:i]
        return NetCDFContainer(fn, mode=mode, **kwargs)[n]
    elif frame is not None:
        return NetCDFContainer(fn, mode=mode, **kwargs)[frame]
    else:
        return NetCDFContainer(fn, mode=mode, **kwargs)
