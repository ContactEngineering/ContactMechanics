#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   Factory.py

@author Till Junge <till.junge@kit.edu>

@date   04 Mar 2015

@brief  Implements a convenient Factory function for Contact System creation

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
from .. import ContactMechanics, SolidMechanics, Surface
from ..Tools import compare_containers
from .Systems import SystemBase
from .Systems import IncompatibleFormulationError
from .Systems import IncompatibleResolutionError


def SystemFactory(substrate, interaction, surface):
    """
    Factory function for contact systems. Checks the compatibility between the
    substrate, interaction method and surface and returns an object of the
    appropriate type to handle it. The returned object is always of a subtype
    of SystemBase.
    Keyword Arguments:
    substrate   -- An instance of HalfSpace. Defines the solid mechanics in
                   the substrate
    interaction -- An instance of Interaction. Defines the contact formulation
    surface     -- An instance of Surface, defines the profile.
    """
    # pylint: disable=invalid-name
    # pylint: disable=no-member
    args = substrate, interaction, surface
    subclasses = list()

    def check_subclasses(base_class, container):
        """
        accumulates a flattened container containing all subclasses of
        base_class
        Parameters:
        base_class -- self-explanatory
        container  -- self-explanatory
        """
        for cls in base_class.__subclasses__():
            check_subclasses(cls, container)
            container.append(cls)

    check_subclasses(SystemBase, subclasses)
    for cls in subclasses:
        print('dealing with class {}'.format(cls))
        if cls.handles(*(type(arg) for arg in args)):
            return cls(*args)
    raise IncompatibleFormulationError(
        ("There is no class that handles the combination of substrates of type"
         " '{}', interactions of type '{}' and surfaces of type '{}'").format(
             *(arg.__class__.__name__ for arg in args)))
