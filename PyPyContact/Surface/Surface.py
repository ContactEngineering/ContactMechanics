#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# @file   Surface.py
#
# @author Till Junge <till.junge@kit.edu>
#
# @date   26 Jan 2015
#
# @brief  Base class for geometric descriptions
#
# @section LICENCE
#
#  Copyright (C) 2015 Till Junge
#
# PyPyContact is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation, either version 3, or (at
# your option) any later version.
#
# PyPyContact is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GNU Emacs; see the file COPYING. If not, write to the
# Free Software Foundation, Inc., 59 Temple Place - Suite 330,
# Boston, MA 02111-1307, USA.
#


class Surface(object):
    """ Base class for geometries. These are used to define height profiles for
         contact problems"""
    class Error(Exception):
        pass
    class NotImplemented(Error):
        pass
    name = 'generic_geom'

    def __init__(self):
        self._resolution = None
        self._dim = None
        self._size = None

    def profile(self, *args, **kwargs):
        """ returns an array of heights
        """
        raise NotImplemented()

    def __add__(self, other):
        return CompoundSurface(self, other)

    def __sub__(self, other):
        return CompoundSurface(self, -1.*other)

    def __mul__(self, other):
        return ScaledSurface(self, other)

    __rmul__ = __mul__

    @property
    def dim(self,):
        """ needs to be testable to make sure that geometry and halfspace are
            compatible
        """
        return self._dim

    @property
    def resolution(self,):
        """ needs to be testable to make sure that geometry and halfspace are
            compatible
        """
        return self._resolution

    @property
    def size(self,):
        """ needs to be testable to make sure that geometry and halfspace are
            compatible
        """
        return self._size

    def save(self, fname, compress=True, *args, **kwargs):
        """ saves the surface as a NumpyTxtSurface
        """
        if compress:
            if not fname.endswith('.gz'):
                fname = fname + ".gz"
        np.savetxt(fname, self.profile(*args, **kwargs))


class ScaledSurface(Surface):
    """ used when geometries are scaled
    """
    name = 'scaled_surface'
    def __init__(self, surf, coeff):
        """
        Keyword Arguments:
        surf  -- Surface to scale
        coeff -- Scaling factor
        """
        assert isinstance(surf, Surface)
        self.surf = surf
        self.coeff = float(coef)

    @property
    def dim(self,):
        """ needs to be testable to make sure that geometry and halfspace are
            compatible
        """
        return self.surf.dim

    @property
    def resolution(self,):
        """ needs to be testable to make sure that geometry and halfspace are
            compatible
        """
        return self.surf.resolution

    def profile(self, args = list(), kwargs = dict()):
        """ Computes the combined profile. Optional *args and **kwargs are
            passed to the surfaces.
        Keyword Arguments:
        args   -- (default list())
        kwargs -- (default dict())
        """
        return self.coeff*self.surf.profile(*args, **kwargs)


class CompoundSurface(Surface):
    """ used when geometries are combined
    """
    name = 'combined_surface'
    def __init__(self, surfA, surfB):
        """ Behaves like a surface that is a sum  of two Surfaces
        Keyword Arguments:
        surfA   -- first surface of the compound
        surfB   -- second surface of the compound
        """
        def combined_val(propA, propB, propname):
            if propA is None:
                return probB
            else:
                if propB is not None:
                    assert propA == propB, \
                        "{} incompatible:{} <-> {}".format(
                            propname, propA, propB)
                return propA

        self._dim = combined_val(surfA.dim, surfB.dim, 'dim')
        self._resulution = combined_val(surfA.resolution,
                                         surfB.resolution, 'resolution')
        self.surfA = surfA
        self.surfB = surfB


    def profile(self, surfA_args = list(), surfA_kwargs = dict(),
            surfB_args = list(), surfB_kwargs = dict()):
        """ Computes the combined profile. Optional *args and **kwargs are
            passed to the surfaces.
        Keyword Arguments:
        surfA_args              -- (default list())
        surfA_kwargs            -- (default dict())
        surfB_args -- (default list())
        surfB_kwargs            -- (default dict())
        """
        return (self.surfA.profile(*surfA_args, **surfA_kwargs) +
                self.surfB.profile(*surfB_args, **surfB_kwargs))

class NumpySurface(Surface):
    """ Dummy surface from a static array
    """
    name = 'surface_from_np_array'
    def __init__(self, profile):
        """
        Keyword Arguments:
        profile -- surface profile
        """
        self.__h = profile
        self.resolution = self.__h.shape
        self.dim = len(self.resolution)

    def profile(self):
        return self.__h

class Sphere(Surface):
    """ Spherical surface. Corresponds to a cylinder in 2D
    """
    name = 'sphere'
    ##def __init__(self, radius, resolution, size
