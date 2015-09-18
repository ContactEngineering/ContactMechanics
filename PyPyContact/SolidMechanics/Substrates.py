#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   Substrate.py

@author Till Junge <till.junge@kit.edu>

@date   26 Jan 2015

@brief  Base class for continuum mechanics models of halfspaces

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


class Substrate(object):
    """ Generic baseclass from which all substate classes derive
    """
    _periodic = None

    class Error(Exception):
        # pylint: disable=missing-docstring
        pass
    name = 'generic_halfspace'

    def spawn_child(self, dummy):
        """ does nothing for most substrates.
        """
        raise self.Error(
            "Only substrates with free boundaries can do this")

    @classmethod
    def is_periodic(cls):
        "non-periodic substrates can use some optimisations"
        if cls._periodic is not None:
            return cls._periodic
        raise cls.Error(
            ("periodicity of Substrate type '{}' ('{}') is not defined"
             "").format(cls.name, cls.__name__))


class ElasticSubstrate(Substrate):
    """ Generic baseclass for elastic substrates
    """
    name = 'generic_elastic_halfspace'
    # Since an elastic substrate essentially defines a Potential, a similar
    # internal structure is chosen

    def __init__(self):
        self.energy = None
        self.force = None

    def compute(self, disp, pot=True, forces=False):
        """
        computes and stores the elastic energy and/or surface forces
        the as function of the surface displacement. Note that forces, not
        surface pressures are expected. This is contrary to most formulations
        in the literature, but convenient in the code (consistency with the
        softWall interaction potentials). This choice may come back to bite me.
        Parameters:
        gap    -- array containing the point-wise gap values
        pot    -- (default True) whether the energy should be evaluated
        forces -- (default False) whether the forces should be evaluated
        """
        self.energy, self.force = self.evaluate(
            disp, pot, forces)

    def evaluate(self, disp, pot=True, forces=False):
        """
        computes and returns the elastic energy and/or surface forces
        as function of the surface displacement. See docstring for 'compute'
        for more details
        Parameters:
        gap    -- array containing the point-wise gap values
        pot    -- (default True) whether the energy should be evaluated
        forces -- (default False) whether the forces should be evaluated
        """
        raise NotImplementedError()


class PlasticSubstrate(Substrate):
    """ Generic baseclass for plastic substrates
    """
    # pylint: disable=too-few-public-methods
    name = 'generic_plastic_halfspace'
