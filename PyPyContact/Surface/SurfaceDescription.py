#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   Surface.py

@author Till Junge <till.junge@kit.edu>

@date   26 Jan 2015

@brief  Base class for geometric descriptions

@section LICENCE

 Copyright (C) 2015 Till Junge

PyPyContact is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation, either version 3, or (at
your option) any later version.

PyPyContact is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with GNU Emacs; see the file COPYING. If not, write to the
Free Software Foundation, Inc., 59 Temple Place - Suite 330,
Boston, MA 02111-1307, USA.
"""

import numpy as np
import abc


class Surface(object, metaclass=abc.ABCMeta):
    """ Base class for geometries. These are used to define height profiles for
         contact problems"""
    class Error(Exception):
        # pylint: disable=missing-docstring
        pass
    name = 'generic_geom'

    def __init__(self, resolution=None, dim=None, size=None,
                 adjustment=0.):
        self._resolution = resolution
        self._dim = dim
        self._size = size
        self.adjustment = adjustment

    def adjust(self):
        """
        shifts surface up or down so that a zero displacement would lead to a
        zero gap
        """
        self.adjustment = self.profile().max()

    def profile(self):
        """ returns an array of possibly adjusted heights
        """
        return self._profile()-self.adjustment

    @abc.abstractmethod
    def _profile(self):
        """ returns an array of heights
        """
        raise NotImplementedError()

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

    def save(self, fname, compress=True):
        """ saves the surface as a NumpyTxtSurface
        """
        if compress:
            if not fname.endswith('.gz'):
                fname = fname + ".gz"
        np.savetxt(fname, self.profile())


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
        super().__init__()
        assert isinstance(surf, Surface)
        self.surf = surf
        self.coeff = float(coeff)

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

    def _profile(self):
        """ Computes the combined profile.
        """
        return self.coeff*self.surf.profile()


class CompoundSurface(Surface):
    """ used when geometries are combined
    """
    name = 'combined_surface'

    def __init__(self, surf_a, surf_b):
        """ Behaves like a surface that is a sum  of two Surfaces
        Keyword Arguments:
        surf_a   -- first surface of the compound
        surf_b   -- second surface of the compound
        """
        super().__init__()

        def combined_val(prop_a, prop_b, propname):
            """
            surfaces can have a fixed or dynamic, adaptive resolution (or other
            attributes). This function assures that -- if this function is
            called for two surfaces with fixed resolutions -- the resolutions
            are identical
            Parameters:
            prop_a   -- field of one surf
            prop_b   -- field of other surf
            propname -- field identifier (for error messages only)
            """
            if prop_a is None:
                return prop_b
            else:
                if prop_b is not None:
                    assert prop_a == prop_b, \
                        "{} incompatible:{} <-> {}".format(
                            propname, prop_a, prop_b)
                return prop_a

        self._dim = combined_val(surf_a.dim, surf_b.dim, 'dim')
        self._resolution = combined_val(surf_a.resolution,
                                        surf_b.resolution, 'resolution')
        self.surf_a = surf_a
        self.surf_b = surf_b

    def _profile(self):
        """ Computes the combined profile
        """
        return (self.surf_a.profile() +
                self.surf_b.profile())


class NumpySurface(Surface):
    """ Dummy surface from a static array
    """
    name = 'surface_from_np_array'

    def __init__(self, profile, size=None):
        """
        Keyword Arguments:
        profile -- surface profile
        """
        self.__h = profile
        super().__init__(resolution=self.__h.shape, dim=len(self.__h.shape),
                         size=size)

    def _profile(self):
        return self.__h


class Sphere(NumpySurface):
    """ Spherical surface. Corresponds to a cylinder in 2D
    """
    name = 'sphere'

    def __init__(self, radius, resolution, size, centre=None, standoff=0):
        """
        Simple shere geometry.
        Parameters:
        radius     -- self-explanatory
        resolution -- self-explanatory
        size       -- self-explanatory
        centre     -- specifies the coordinates (in lenght units, not pixels).
                      by default, the sphere is centred in the surface
        standoff   -- when using interaction forces with ranges of the order
                      the radius, you might want to set the surface outside of
                      the spere to far away, maybe even pay the price of inf,
                      if your interaction has no cutoff
        """
        # pylint: disable=invalid-name
        if not hasattr(resolution, "__iter__"):
            resolution = (resolution, )
        dim = len(resolution)
        if not hasattr(size, "__iter__"):
            size = (size, )
        if centre is None:
            centre = np.array(size)*.5
        if not hasattr(centre, "__iter__"):
            centre = (centre, )

        if dim == 1:
            r2 = (np.arange(resolution[0], dtype=float) *
                  size[0] / resolution[0] - centre[0])**2
        elif dim == 2:
            rx2 = ((np.arange(resolution[0], dtype=float) *
                    size[0] / resolution[0] - centre[0])**2).reshape((-1, 1))
            ry2 = (np.arange(resolution[1], dtype=float) *
                   size[1] / resolution[1] - centre[1])**2
            r2 = rx2 + ry2
        radius2 = radius**2  # avoid nans for small radiio
        outside = r2 > radius2
        r2[outside] = radius2
        h = np.sqrt(radius2 - r2)-radius
        h[outside] -= standoff
        super().__init__(h)
        self._size = size
        self._centre = centre

    @property
    def centre(self):
        "returns the coordinates of the sphere's (or cylinder)'s centre"
        return self._centre
