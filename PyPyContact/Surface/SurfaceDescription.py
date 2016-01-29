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
#from ..Tools.SurfaceAnalysis import compute_rms_slope

class Surface(object, metaclass=abc.ABCMeta):
    """ Base class for geometries. These are used to define height profiles for
         contact problems"""
    class Error(Exception):
        # pylint: disable=missing-docstring
        pass
    name = 'generic_geom'

    def __init__(self, resolution=None, dim=None, size=None, adjustment=0.):
        self._resolution = resolution
        self._dim = dim
        self._size = size
        self.adjustment = adjustment

    def __getstate__(self):
        """ is called and the returned object is pickled as the contents for
            the instance
        """
        state = (self._resolution, self._dim, self._size, self.adjustment)
        return state

    def __setstate__(self, state):
        """ Upon unpickling, it is called with the unpickled state
        Keyword Arguments:
        state -- result of __getstate__
        """
        (self._resolution, self._dim, self._size, self.adjustment) = state

    def compute_rms_height(self):
        "computes the rms height fluctuation of the surface"
        delta = self.profile()
        delta -= delta.mean()
        return np.sqrt((delta**2).mean())

    def compute_rms_height_q_space(self):
        """
        computes the rms height fluctuation of the surface in the
        frequency domain
        """
        delta = self.profile()
        delta -= delta.mean()
        area = np.prod(self.size)
        nb_pts = np.prod(self.resolution)
        H = area/nb_pts*np.fft.fftn(delta)
        return 1/area*np.sqrt((np.conj(H)*H).sum().real)

    def compute_rms_slope(self):
        "computes the rms height gradient fluctuation of the surface"
        return compute_rms_slope(self.profile(), resolution=self.resolution,
                                 size=self.size, dim=self.dim)

    def compute_rms_slope_q_space(self):
        """
        taken from roughness in pycontact
        """
        nx, ny = self.resolution
        sx, sy = self.size
        qx = np.arange(nx, dtype=np.float64)
        qx = np.where(qx <= nx/2, 2*np.pi*qx/sx, 2*np.pi*(nx-qx)/sx)
        qy = np.arange(ny, dtype=np.float64)
        qy = np.where(qy <= ny/2, 2*np.pi*qy/sy, 2*np.pi*(ny-qy)/sy)
        q  = np.sqrt( (qx*qx).reshape(-1, 1) + (qy*qy).reshape(1, -1) )

        h_q = np.fft.fft2(self.profile())
        return np.sqrt(
            np.mean(q**2 * h_q*np.conj(h_q)).real/(float(self.profile().shape[0])*float(self.profile().shape[1])))


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

    def __getitem__(self, index):
        return self._profile()[index]-self.adjustment

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
    shape = resolution

    def set_size(self, size, sy=None):
        """ set the size of the surface """
        if sy is not None:
            size = (size, sy)
        self._size = size

    @property
    def size(self,):
        """ needs to be testable to make sure that geometry and halfspace are
            compatible
        """
        return self._size

    @size.setter
    def size(self, size):
        """ set the size of the surface """
        if not hasattr(size, "__iter__"):
            size = (size, )
        else:
            size = tuple(size)
        if len(size) != self.dim:
            raise self.Error(
                ("The dimension of this surface is {}, you have specified an "
                 "incompatible size of dimension {} ({}).").format(
                     self.dim, len(size), size))
        self._size = size

    def save(self, fname, compress=True):
        """ saves the surface as a NumpyTxtSurface. Warning: This only saves
            the profile; the size is not contained in the file
        """
        if compress:
            if not fname.endswith('.gz'):
                fname = fname + ".gz"
        np.savetxt(fname, self.profile())

    def estimate_laplacian(self, coords):
        """
        estimate the local laplacian at coords by finite differences
        Keyword Arguments:
        coords --
        """
        laplacian = 0.
        for i in range(self.dim):
            pixel_size = self.size[i] / self.resolution[i]
            coord = coords[i]
            if coord == 0:
                delta = 1
            elif coord == self.resolution[i]-1:
                delta = -1
            else:
                delta = 0
            irange = (coords[i]-1+delta, coords[i]+delta, coords[i]+1+delta)
            fun_val = np.zeros(len(irange))
            for j in range(len(irange)):
                coord_copy = list(coords)
                coord_copy[i] = irange[j]
                try:
                    fun_val[j] = self.profile()[tuple(coord_copy)]
                except IndexError as err:
                    raise IndexError(
                        ("{}:\ncoords = {}, i = {}, j = {}, irange = {}, "
                         "coord_copy = {}").format(
                             err, coords, i, j, irange, coord_copy))  # nopep8
            laplacian += (fun_val[0] + fun_val[2] - 2*fun_val[1])/pixel_size**2
        return laplacian


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
    shape = resolution

    @property
    def size(self,):
        """ needs to be testable to make sure that geometry and halfspace are
            compatible
        """
        return self.surf.size

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
        self._size = combined_val(surf_a.size,
                                  surf_b.size, 'size')
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

    def __getstate__(self):
        """ is called and the returned object is pickled as the contents for
            the instance
        """
        state = (super().__getstate__(), self.__h)
        return state

    def __setstate__(self, state):
        """ Upon unpickling, it is called with the unpickled state
        Keyword Arguments:
        state -- result of __getstate__
        """
        super().__setstate__(state[0])
        self.__h = state[1]

class Sphere(NumpySurface):
    """ Spherical surface. Corresponds to a cylinder in 2D
    """
    name = 'sphere'

    def __init__(self, radius, resolution, size, centre=None, standoff=0, periodic=False):
        """
        Simple shere geometry.
        Parameters:
        radius     -- self-explanatory
        resolution -- self-explanatory
        size       -- self-explanatory
        centre     -- specifies the coordinates (in length units, not pixels).
                      by default, the sphere is centred in the surface
        standoff   -- when using interaction forces with ranges of the order
                      the radius, you might want to set the surface outside of
                      the spere to far away, maybe even pay the price of inf,
                      if your interaction has no cutoff
        periodic   -- whether the sphere can wrap around. tricky for large spheres
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

        if not periodic:
            def get_r(res, size, centre):
                return np.linspace(-centre, size-centre, res, endpoint=False)
        else:
            def get_r(res, size, centre):
                return np.linspace(-centre +   size/2,
                                   -centre + 3*size/2,
                                   res, endpoint=False)%size - size/2

        if dim == 1:
            r2 = get_r(resolution[0],
                       size[0],
                       centre[0])**2
        elif dim == 2:
            rx2 = (get_r(resolution[0],
                         size[0],
                         centre[0])**2).reshape((-1, 1))
            ry2 = (get_r(resolution[1],
                         size[1],
                         centre[1]))**2
            r2 = rx2 + ry2
        else:
            raise Exception(
                ("Problem has to be 1- or 2-dimensional. "
                 "Yours is {}-dimensional").format(dim))
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
