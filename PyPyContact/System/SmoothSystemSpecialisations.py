#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# @file   SmoothSystemSpecialisations.py
#
# @author Till Junge <till.junge@kit.edu>
#
# @date   20 Feb 2015
#
# @brief  implements the periodic and non-periodic smooth contact systems
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
from .System import SmoothContactSystem
from ..Surface import NumpySurface


## convenient container for storing correspondences betwees small and large system
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
    def __init__(self, substrate, interaction, surface, margin=4):
        """ Represents a contact problem
        Keyword Arguments:
        substrate   -- An instance of HalfSpace. Defines the solid mechanics in
                       the substrate
        interaction -- An instance of Interaction. Defines the contact formulation
        surface     -- An instance of Surface, defines the profile.
        margin      -- (default 4) safety margin (in pixels) around the initial
                       contact area bounding box
        """
        super().__init__(substrate, interaction. surface)
        ## create empty encapsulated syste
        self.babushka = None
        self.margin = margin

    def objective(self, offset, disp0, gradient=False):
        """
        See super().objective for general description this method's purpose.
        Difference for this class wrt 'dumb' systems:
        Needs an initial guess for the displacement field in order to estimate
        the contact area. returns both the objective and the adapted ininial
        guess as a tuple
        Keyword Arguments:
        offset   -- determines indentation depth
        gradient -- (default False) whether the gradient is supposed to be used
        disp0    -- initial guess for displacement field.
        """
        ## this class needs to remember its offset since the evaluate method
        ## does not accept it as argument anymore
        self.__offset = offset 
        
        gap = self.computeGap(disp0, offset)
        contact = np.argwhere(gap < self.interaction.r_c)
        self.offset = tuple(bnd - self.margin for bnd in  np.min(contact, 0))
        sm_res = tuple(bnd - self.margin - offset[i] for i, bnd in
                       enumerate(np.max(contact, 0)))

        self.computeBabushkaBounds(self.offset, sm_res)
        sm_disp0 = self._getBabushkaArray(self.surface.profile())

        sm_substrate = self.substrate.spawnChild(sm_res)
        sm_surface = NumpySurface(sm_disp0)
        self.babushka = SmoothContactSystem(
            sm_substrate, self.interaction, sm_surface)

        return self.babushka.objective(offset, gradient), sm_disp0.copy()

    def evaluate(self):
        raise NotImplementedError()
        self.substrate.force = self._getFullArray(self.babushka.substrate.force)
        self.interaction.force = self._getFullArray(
            self.babushka.interaction.force)
        self.energy = self.babushka.energy

        self.force = self.substrate.force.copy()
        if self.dim == 1:
            self.force[:self.resolution[0]] -= self.interaction.force
        else:
            self.force[:self.resolution[0], :self.resolution[1]] -= \
              self.interaction.force
        disp = self.substrate.evaluateDisp(substrate_force)
        return self.energy, self.force, disp


    def computeBabushkaBounds(self, offset, babushka_resolution):
        def boundary_generator():
          sm_res = babushka_resolution
          lg_res = self.resolution
          for i in (0,1):
            for j in (0,1):
             sm_slice = tuple(slice(i*sm_res[0], (i+1)*sm_res[0]),
                              slice(j*sm_res[1], (j+1)*sm_res[1]))
             lg_slice = tuple(
              slice(i*lg_res[0]+self.offset[0], (i+1)*lg_res[0]+self.offset[0]),
              slice(j*lg_res[1]+self.offset[1], (j+1)*lg_res[1]+self.offset[1]))
              yield(BndSet(large=lg_slice, small=sm_slice))
        self.bounds = tuple((bnd for bnd in boundary_generator()))

    def _getBabushkaArray(self, full_array, babushka_array=None):
        def computational_resolution():
            if babushka_array is None:
                babushka_array = np.zeros(
                    self.babushka.substrate.computational_resolution)
            for bnd in self.bounds:
                babushka_array[bnd.small] = full_array[bnd.large]
            return babushka_array
        def normal_resolution():
            if babushka_array is None:
                babushka_array = np.zeros(self.babushka.resolution)
            bnd = self.bounds[0]
            babushka_array[bnd.small] = full_array[bnd.large]
            return babushka_array
        if full_array.shape = self.resolution:
            return normal_resolution()
        else:
            return computational_resolution()

    def _getFullArray(self, babushka_array, full_array=None):
        def computational_resolution():
            if full_array is None:
                full_array = np.zeros(
                    self.substrate.computational_resolution)
            for bnd in self.bounds:
                full_array[bnd.large] = babushka_array[bnd.small]
            return full_array
        def normal_resolution():
            if full_array is None:
                full_array = np.zeros(self.resolution)
            bnd = self.bounds[0]
            full_array[bnd.large] = babushka_array[bnd.small]
            return full_array
        if babushka_array.shape = self.babushka.resolution:
            return normal_resolution()
        else:
            return computational_resolution()
