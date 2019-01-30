#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   Factory.py

@author Till Junge <till.junge@kit.edu>

@date   04 Mar 2015

@brief  Implements a convenient Factory function for Contact System creation

@section LICENCE

Copyright 2015-2017 Till Junge, Lars Pastewka

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from .. import ContactMechanics, SolidMechanics, Topography
from ..Tools import compare_containers
from .Systems import SystemBase
from .Systems import IncompatibleFormulationError
from .Systems import IncompatibleResolutionError


def make_system(substrate, interaction, surface):
    """
    Factory function for contact systems. Checks the compatibility between the
    substrate, interaction method and surface and returns an object of the
    appropriate type to handle it. The returned object is always of a subtype
    of SystemBase.
    Keyword Arguments:
    substrate   -- An instance of HalfSpace. Defines the solid mechanics in
                   the substrate
    interaction -- An instance of Interaction. Defines the contact formulation
    surface     -- An instance of Topography, defines the profile.
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
        if cls.handles(*(type(arg) for arg in args)):
            return cls(*args)
    raise IncompatibleFormulationError(
        ("There is no class that handles the combination of substrates of type"
         " '{}', interactions of type '{}' and surfaces of type '{}'").format(
             *(arg.__class__.__name__ for arg in args)))
