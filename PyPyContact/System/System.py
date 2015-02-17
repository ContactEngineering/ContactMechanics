#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# @file   System.py
#
# @author Till Junge <till.junge@kit.edu>
#
# @date   26 Jan 2015
#
# @brief  Defines the interface for PyPyContact systems
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
# Free Software Foundation, Inc., 59 Temple Place - Su ite 330,
# Boston, MA 02111-1307, USA.
#
import numpy as np

from .. import ContactMechanics, SolidMechanics, Surface
from ..Tools import compare_containers

class SystemBase(object):
    def __init__(self, substrate, interaction, surface):
        """ Represents a contact problem
        Keyword Arguments:
        substrate   -- An instance of HalfSpace. Defines the solid mechanics in
                       the substrate
        interaction -- An instance of Interaction. Defines the contact formulation
        surface     -- An instance of Surface, defines the profile.
        """

    @staticmethod
    def handles(substrate_type, interaction_type, surface_type):
        """
        returns whether this class (in practice a subclass) handles this
        combination of types
        Keyword Arguments:
        substrate_type   -- self-explanatory
        interaction_type -- self-explanatory
        surface_type     -- self-explanatory
        """
        return False

    def computeGap(self, disp, offset, *profile_args, **profile_kwargs):
        """
        """
        if self.dim == 1:
            return (offset - disp[:self.resolution[0]] -
                    self.surface.profile(*profile_args, **profile_kwargs))
        return (offset - disp[:self.resolution[0], :self.resolution[1]] -
                self.surface.profile(*profile_args, **profile_kwargs))

class SmoothContactSystem(SystemBase):
    def __init__(self, substrate, interaction, surface):
        """ Represents a contact problem
        Keyword Arguments:
        substrate   -- An instance of HalfSpace. Defines the solid mechanics in
                       the substrate
        interaction -- An instance of Interaction. Defines the contact formulation
        surface     -- An instance of Surface, defines the profile.
        """
        self.substrate = substrate
        self.interaction = interaction
        self.surface = surface
        if not compare_containers(surface.resolution,substrate.resolution):
            raise IncompatibleResolutionError(
                ("the substrate ({}) and the surface ({}) have incompatible "
                 "resolutions.").format(
                     substrate.resolution, surface.resolution))
        self.dim = len(self.substrate.resolution)
    @property
    def resolution(self):
        return self.surface.resolution

    @staticmethod
    def handles(substrate_type, interaction_type, surface_type):
        is_ok = True
        ## any type of substrate formulation should do
        is_ok &= issubclass(substrate_type,
                            SolidMechanics.Substrate)
        ## only soft interactions allowed
        is_ok &= issubclass(interaction_type,
                            ContactMechanics.SoftWall)

        ## any surface should do
        is_ok &= issubclass(surface_type,
                            Surface.Surface)
        return is_ok

    def evaluate(self, disp, offset, pot=True, forces=False):
        """
        Compute the energies and forces in the system for a given displacement
        field
        """
        ## attention: the substrate may have a higher resolution than the gap
        ## and the interaction (e.g. FreeElasticHalfSpace)
        self.gap = self.computeGap(disp, offset)
        self.interaction.compute(self.gap, pot, forces)
        self.substrate.compute(disp, pot, forces)

        ## attention: the gradient of the interaction has the wrong sign,
        ## because the derivative of the gap with respect to displacement
        ##introduces a -1 factor. Hence the minus sign in the 'sum' of forces:
        self.energy = (self.interaction.energy + self.substrate.energy
                       if pot else None)
        if forces:
            self.force = self.substrate.force
            if self.dim == 1:
                self.force[:self.resolution[0]] -= self.interaction.force
            else:
                self.force[:self.resolution[0], :self.resolution[1]] -= \
                  self.interaction.force
        else:
            self.force = None

        return (self.energy, self.force)

    def objective(self, offset, gradient=False):
        """
        This helper method exposes a scipy.optimize-friendly interface to the
        evaluate() method. Use this for optimization purposes, it makes sure
        that the shape of disp is maintained and lets you set the offset and
        'forces' flag without using scipy's cumbersome argument passing
        interface. Returns a function of only disp
        Keyword Arguments:
        offset      -- determines indentation depth
        forces      -- (default False) whether the gradient is supposed to be used
        """
        res = self.substrate.computational_resolution
        if gradient:
            def fun(disp):
                print("been called wit an input of shape {}".format(disp.shape))
                self.evaluate(
                    disp.reshape(res), offset, forces=True)
                return (self.energy, -self.force.reshape(-1))
            return fun
        else:
            return lambda disp: self.evaluate(
                disp.reshape(res), offset, forces=False)[0]

    def callback(self, force=False):
        counter = 0
        if force:
            def fun(vdisp):
                nonlocal counter
                counter +=1
                print("at it {}, e = {}, |f| = {}".format(
                    counter, self.energy, np.linalg.norm(np.ravel(self.force))))
        else:
            def fun(vdisp):
                nonlocal counter
                counter +=1
                print("at it {}, e = {}".format(
                    counter, self.energy))
        return fun

class IncompatibleFormulationError(Exception):
    pass
class IncompatibleResolutionError(Exception):
    pass

def System(substrate, interaction, surface):
    """
    Factory function for contact systems. Checks the compatibility between the
    substrate, interaction method and surface and returns an object of the
    appropriate type to handle it. The returned object is always of a subtype of
    SystemBase.
    Keyword Arguments:
    substrate   -- An instance of HalfSpace. Defines the solid mechanics in
                   the substrate
    interaction -- An instance of Interaction. Defines the contact formulation
    surface     -- An instance of Surface, defines the profile.
    """
    args = substrate, interaction, surface
    for cls in SystemBase.__subclasses__():
        if cls.handles(*(type(arg) for arg in args)):
            return cls(*args)
    raise IncompatibleFormulationError(
        ("There is no class that handles the combination of substrates of type "
         "'{}', interactions of type '{}' and surfaces of type '{}'").format(
             *(arg.__class__.__name__ for arg in args)))


if __name__ == "__main__":
    sb = SystemBase( 1, 5 ,4)
    System(1, 5, 4)
