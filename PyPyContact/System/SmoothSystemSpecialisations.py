#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   SmoothSystemSpecialisations.py

@author Till Junge <till.junge@kit.edu>

@date   20 Feb 2015

@brief  implements the periodic and non-periodic smooth contact systems

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

from collections import namedtuple
import numpy as np

from .Systems import SmoothContactSystem
from ..Surface import NumpySurface
from .. import ContactMechanics, SolidMechanics, Surface


# convenient container for storing correspondences betwees small and large
# system
BndSet = namedtuple('BndSet', ('large', 'small'))


class FastSmoothContactSystem(SmoothContactSystem):
    """
    This proxy class tries to take advantage of the system-size independence of
    non-periodic FFT-solved systems by determining the required minimum system
    size, encapsulating a SmoothContactSystem of that size and run it almost
    transparently instead of the full system.
    It's almost transparent, because by its nature, the small system does not
    compute displacements everywhere the large system exists. Therefore, for
    full-domain output, an extra evaluation step must be taken. But since using
    the small system significantly increases optimisation speed, this tradeoff
    seems to be well worth the trouble.
    """
    # declare that this class is only a proxy
    _proxyclass = True
    class FreeBoundaryError(Exception):
        """
        called when the supplied system cannot be computed with the current
        constraints
        """
        pass

    def __init__(self, substrate, interaction, surface, margin=1):
        """ Represents a contact problem
        Keyword Arguments:
        substrate   -- An instance of HalfSpace. Defines the solid mechanics in
                       the substrate
        interaction -- An instance of Interaction. Defines the contact
                       formulation
        surface     -- An instance of Surface, defines the profile.
        margin      -- (default 4) safety margin (in pixels) around the initial
                       contact area bounding box
        """
        super().__init__(substrate, interaction, surface)
        # create empty encapsulated system
        self.babushka = None
        self.margin = margin
        self.bounds = None
        # coordinates of the bottom-right corner of the small scale domain in
        # the large scale domain
        self.__babushka_offset = None
        # This is the verticas offset between surface and substrate. Determined
        # indentation depth. This class needs to keep track of the offsets for
        # which its babushka has been evaluated, in order to have the ability
        # to interpolate the babushka-results onto the domain
        self.offset = None
        self.energy = None
        self.force = None

    def shape_minimisation_input(self, in_array):
        """
        For minimisation of smart systems, the initial guess array (e.g.
        displacement) may have a non-intuitive shape and size (The problem size
        may be decreased, as for free, non-periodic systems, or increased as
        with augmented-lagrangian-type issues). Use the output of this function
        as argument x0 for scipy minimisation functions. Also, if you initial
        guess has a shape that makes no sense, this will tell you before you
        get caught in debugging scipy-code

        Arguments:
        in_array -- array with the initial guess. has the intuitive shape you
                    think it has
        """
        if np.prod(self.substrate.computational_resolution) == in_array.size:
            return self._get_babushka_array(in_array).reshape(-1)
        elif (np.prod(self.babushka.substrate.computational_resolution) ==
              in_array.size):
            return in_array.reshape(-1)
        raise IncompatibleResolutionError()

    @staticmethod
    def handles(substrate_type, interaction_type, surface_type):
        is_ok = True
        # any periodic type of substrate formulation should do
        is_ok &= issubclass(substrate_type,
                            SolidMechanics.Substrate)
        if is_ok:
            is_ok &= ~substrate_type.is_periodic()
        # only soft interactions allowed
        is_ok &= issubclass(interaction_type,
                            ContactMechanics.SoftWall)

        # any surface should do
        is_ok &= issubclass(surface_type,
                            Surface.Surface)
        return is_ok

    def objective(self, offset, disp0=None, gradient=False):
        """
        See super().objective for general description this method's purpose.
        Difference for this class wrt 'dumb' systems:
        Needs an initial guess for the displacement field in order to estimate
        the contact area. returns both the objective and the adapted ininial
        guess as a tuple
        Keyword Arguments:
        offset   -- determines indentation depth
        gradient -- (default False) whether the gradient is supposed to be used
        disp0    -- (default np.zeros) initial guess for displacement field.
                    if not chosen appropriately, results may be unreliable
        """
        # pylint: disable=arguments-differ
        # this class needs to remember its offset since the evaluate method
        # does not accept it as argument anymore
        self.offset = offset
        if disp0 is None:
            disp0 = np.zeros(self.substrate.computational_resolution)
        gap = self.compute_gap(disp0, offset)
        contact = np.argwhere(gap < self.interaction.r_c)
        # Lower bounds by dimension of the indices of contacting cells
        bnd_lo = np.min(contact, 0)
        # Upper bounds by dimension of the indices of contacting cells
        bnd_up = np.max(contact, 0)
        print(bnd_up)

        self.__babushka_offset = tuple(bnd - self.margin for bnd in bnd_lo)
        sm_res = tuple((hi-lo + 2*self.margin for (hi, lo) in
                        zip(bnd_up, bnd_lo)))
        print(sm_res)
        if any(bnd < 0 for bnd in self.__babushka_offset):
            raise self.FreeBoundaryError(
                ("With the current margin of {}, the system overlaps the lower"
                 " bounds by {}").format(self.margin, self.__babushka_offset))
        if any(res + self.__babushka_offset[i] > self.resolution[i] for i, res
               in enumerate(sm_res)):
            raise self.FreeBoundaryError(
                ("With the current margin of {}, the system overlaps the upper"
                 " bounds by {}").format(
                     self.margin,
                     tuple(self.__babushka_offset[i] + res - self.resolution[i]
                           for i, res in enumerate(sm_res))))

        self.compute_babushka_bounds(sm_res)
        sm_surf = self._get_babushka_array(self.surface.profile(),
                                            np.zeros(sm_res))
        cntct = gap < self.interaction.r_c
        sm_cntct = self._get_babushka_array(cntct,
                                            np.zeros(sm_res))
        #####import matplotlib.pyplot as plt
        #####plt.figure()
        #####plt.spy(cntct)
        #####plt.figure()
        #####plt.spy(sm_cntct)
        #####plt.show()



        sm_substrate = self.substrate.spawn_child(sm_res)
        sm_surface = NumpySurface(sm_surf)
        self.babushka = SmoothContactSystem(
            sm_substrate, self.interaction, sm_surface)

        return self.babushka.objective(offset, gradient)

    def callback(self, force=False):
        return self.babushka.callback(force)

    def evaluate(self, disp, offset, pot=True, forces=False):
        raise Exception(
            "This proxy-class cannot be evaluated. If you do not understand this, use the base-class instead")

    def deproxyfied(self):
        self.substrate.force = self._get_full_array(
            self.babushka.substrate.force)
        self.interaction.force = self._get_full_array(
            self.babushka.interaction.force)
        self.energy = self.babushka.energy

        self.force = self.substrate.force.copy()
        if self.dim == 1:
            self.force[:self.resolution[0]] -= self.interaction.force
        else:
            self.force[:self.resolution[0], :self.resolution[1]] -= \
              self.interaction.force
        disp = self.substrate.evaluate_disp(self.substrate.force)
        return self.energy, self.force, disp

    def compute_babushka_bounds(self, babushka_resolution):
        """
        returns a list of tuples that contain the equivalent slices in the
        small and the large array. It differentiates between resolution and
        computational_resolution.
        Parameters:
        babushka_resolution -- resolution of smaller scale
        """
        def boundary_generator():
            """
            computes slices for the boundaries. helps translating between large
            and small arrays using copy-less ndarray views
            """
            sm_res = babushka_resolution
            lg_res = self.resolution
            for i in (0, 1):
                for j in (0, 1):
                    sm_slice = tuple((slice(i*sm_res[0], (i+1)*sm_res[0]),
                                      slice(j*sm_res[1], (j+1)*sm_res[1])))
                    lg_slice = tuple((
                        slice(i*lg_res[0]+self.__babushka_offset[0],
                              i*lg_res[0]+sm_res[0]+self.__babushka_offset[0]),
                        slice(j*lg_res[1]+self.__babushka_offset[1],
                              j*lg_res[1]+sm_res[1]+self.__babushka_offset[1])))
                    yield BndSet(large=lg_slice, small=sm_slice)
        self.bounds = tuple((bnd for bnd in boundary_generator()))

    def _get_babushka_array(self, full_array, babushka_array=None):
        """
        returns the equivalent small-scale array representation of a given
        large-scale array. In the case of computational_resolution arrays, this
        is a copy. Else a view.
        Parameters:
        full_array     -- large-scale input array
        babushka_array -- optional small-scale output array to overwrite
        """
        # pylint: disable=unused-argument
        def computational_resolution():
            "used when arrays correspond to the substrate"
            nonlocal babushka_array
            if babushka_array is None:
                babushka_array = np.zeros(
                    self.babushka.substrate.computational_resolution)
            for bnd in self.bounds:
                babushka_array[bnd.small] = full_array[bnd.large]
            return babushka_array

        def normal_resolution():
            "used when arrays correspond to the interaction or the surface"
            nonlocal babushka_array
            if babushka_array is None:
                babushka_array = np.zeros(self.babushka.resolution)
            bnd = self.bounds[0]
            babushka_array[bnd.small] = full_array[bnd.large]
            return babushka_array
        if full_array.shape == self.resolution:
            return normal_resolution()
        else:
            return computational_resolution()

    def _get_full_array(self, babushka_array, full_array=None):
        """
        returns the equivalent large-scale array representation of a given
        Small-scale array.
        Parameters:
        full_array     -- optional large-scale output array to overwrite
        babushka_array -- small-scale input array
        """
        # pylint: disable=unused-argument
        def computational_resolution():
            "used when arrays correspond to the substrate"
            nonlocal full_array
            if full_array is None:
                full_array = np.zeros(
                    self.substrate.computational_resolution)
            for bnd in self.bounds:
                full_array[bnd.large] = babushka_array[bnd.small]
            return full_array

        def normal_resolution():
            "used when arrays correspond to the interaction or the surface"
            nonlocal full_array
            if full_array is None:
                full_array = np.zeros(self.resolution)
            bnd = self.bounds[0]
            full_array[bnd.large] = babushka_array[bnd.small]
            return full_array
        if babushka_array.shape == self.babushka.resolution:
            return normal_resolution()
        else:
            return computational_resolution()
