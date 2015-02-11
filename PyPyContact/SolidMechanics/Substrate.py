#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# @file   Substrate.py
#
# @author Till Junge <till.junge@kit.edu>
#
# @date   26 Jan 2015
#
# @brief  Base class for continuum mechanics models of halfspaces
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

class Substrate(object):
    """ Generic baseclass from which all substate classes derive
    """
    class Error(Exception):pass
    name = 'generic_halfspace'

    def evaluateFromDisp(self, disp):
        force = self.evaluateForce(disp)
        return self.evaluateElasticEnergy(force, disp), force
    def evaluateFromForce(self, force):
        disp = self.evaluateDisp(force)
        return self.evaluateElasticEnergy(force, disp), force

class ElasticSubstrate(Substrate):
    """ Generic baseclass for elastic substrates
    """
    name = 'generic_elastic_halfspace'
    ## Since an elastic substrate essentially defines a Potential, a similar
    ## internal structure is chosen

    def compute(self, disp, pot=True, forces=False):
        self.energy, self.force = self.evaluate(
            disp, pot, forces)

class PlasticSubstrate(Substrate):
    """ Generic baseclass for plastic substrates
    """
    name = 'generic_plastic_halfspace'
