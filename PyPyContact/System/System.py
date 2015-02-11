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
        return (offset - self.surface.profile(
            *profile_args, **profile_kwargs) - disp)

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
        gap = self.computeGap(disp, offset)
        self.interaction.compute(gap, pot, forces)
        self.substrate.compute(disp, pot, forces)
        return ((self.interaction.energy + self.substrate.energy if pot else None),
                (self.interaction.force + self.substrate.force if forces else None))



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
