#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# @file   FFTElasticHalfSpace.py
#
# @author Till Junge <till.junge@kit.edu>
#
# @date   26 Jan 2015
#
# @brief  Imprement the FFT-based elasticity solver of pycontact
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

import multiprocessing
import numpy as np
from scipy.fftpack import fftn, ifftn

from .Substrate import ElasticSubstrate

nb_cores = multiprocessing.cpu_count()

class FFTElasticHalfSpace(ElasticSubstrate):
    """ Uses the FFT to solve the displacements and stresses in an elastic
        Halfspace due to a given array of point forces. This halfspace implemen-
        tation cheats somewhat: since a net pressure would result in infinite
        displacement, the first term of the FFT is systematically dropped.
        The implementation follows the description in Stanley & Kato J. Tribol.
        119(3), 481-485 (Jul 01, 1997)
    """
    name = "fft_elastic_halfspace"
    def __init__(self, resolution, young, size=2*np.pi):
        """
        Keyword Arguments:
        resolution -- Tuple containing number of points in spatial directions.
                      The length of the tuple determines the spatial dimension
                      of the problem.
        young      -- Equiv. Young's modulus E'
                      1/E' = (i-ν_1**2)/E'_1 + (i-ν_2**2)/E'_2
        size       -- (default 2π) domain size. For multidimensional problems,
                      a tuple can be provided to specify the lenths per
                      dimension. If the tuple has less entries than dimensions,
                      the last value in repeated.
        """
        if not hasattr(resolution, "__iter__"):
            resolution = (resolution, )
        if not hasattr(size, "__iter__"):
            size = (size, )
        self.__dim = len(resolution)
        if self.dim not in (1, 2):
            raise self.Error(
                ("Dimension of this problem is {}. Only 1 and 2-dimensional "
                 "problems are supported").format(self.dim))
        self.resolution = resolution
        tmpsize = list()
        for i in range(self.dim):
            tmpsize.append(size[min(i, len(size)-1)])
        self.size = tuple(tmpsize)
        self.nb_pts = np.prod(self.resolution)
        self.area_per_pt = np.prod(self.size)/self.nb_pts

        self.young = young

        self.__computeFourierCoeffs()
        self.__computeIFourierCoeffs()

    @property
    def dim(self, ):
        return self.__dim

    def __repr__(self):
        dims = 'x', 'y', 'z'
        size_str = ', '.join('{}: {}({})'.format(dim, size, resolution) for
                             dim, size, resolution in zip(dims, self.size,
                                                          self.resolution))
        return ("{0.dim}-dimensional halfspace '{0.name}', size(resolution) in "
                "{1}, E' = {0.young}").format(self, size_str)

    def __computeFourierCoeffs(self):
        """Compute the weights w relating fft(displacement) to fft(pressure):
           fft(u) = w*fft(p), see (6) Stanley & Kato J. Tribol. 119(3), 481-485
           (Jul 01, 1997)
        """
        facts = np.zeros(self.resolution)
        if self.dim == 1:
            for index in range(2, self.resolution[0]//2+2):
                facts[-index+1] = facts[index - 1] = 2./(self.young*index)
        if self.dim == 2:
            for m in range(1, self.resolution[0]//2+2):
                for n in range(1, self.resolution[1]//2+2):
                    facts[-m+1, -n+1] = facts[-m+1, n-1] = facts[m-1, -n+1] = \
                      facts[m-1, n-1] = 2./(self.young*(m**2+n**2)**.5)
            facts[0, 0] = 0
        self.weights = facts

    def __computeIFourierCoeffs(self):
        """Compute the weights w relating fft(displacement) to fft(pressure):
           fft(u) = w*fft(p), see (6) Stanley & Kato J. Tribol. 119(3), 481-485
           (Jul 01, 1997)
        """
        self.iweights = np.zeros_like(self.weights)
        self.iweights[self.weights != 0] = 1./self.weights[self.weights != 0]

    def evaluateDisp(self, forces):
        """ Computes the displacement due to a given force array
        Keyword Arguments:
        forces   -- a numpy array containing point forces (*not* pressures)
        """
        if forces.shape != self.resolution:
            raise self.Error(
                ("force array has a different shape ({0}) than this halfspace's"
                 " resolution ({1})").format(forces.shape, self.resolution))
        return ifftn(self.weights * fftn(forces)).real/self.area_per_pt

    def evaluateForce(self, disp):
        """ Computes the forcedue to a given displacement array
        Keyword Arguments:
        disp   -- a numpy array containing point displacements
        """
        if disp.shape != self.resolution:
            raise self.Error(
                ("force array has a different shape ({0}) than this halfspace's"
                 " resolution ({1})").format(disp.shape, self.resolution))
        return ifftn(self.iweights*fftn(disp)).real*self.area_per_pt


    def evaluateKDisp(self, forces):
        """ Computes the K-space displacement due to a given force array
        Keyword Arguments:
        forces   -- a numpy array containing point forces (*not* pressures)
        """
        if forces.shape != self.resolution:
            raise self.Error(
                ("force array has a different shape ({0}) than this halfspace's"
                 " resolution ({1})").format(forces.shape, self.resolution))
        return self.weights * fftn(forces)/self.area_per_pt

    def evaluateKForce(self, disp):
        """ Computes the K-space forces due to a given displacement array
        Keyword Arguments:
        disp   -- a numpy array containing point displacements
        """
        if disp.shape != self.resolution:
            raise self.Error(
                ("force array has a different shape ({0}) than this halfspace's"
                 " resolution ({1})").format(disp.shape, self.resolution))
        return self.iweights*fftn(disp)*self.area_per_pt

    def evaluateElasticEnergy(self, forces, disp):
        return .5*np.dot(np.ravel(disp), np.ravel(forces))

    def evaluateElasticEnergyKspace(self, Kforces, Kdisp):
        return .5*np.dot(np.ravel(Kdisp), np.ravel(np.conj(Kforces))).real/self.nb_pts

    def evaluate(self, disp, pot=True, forces=False):
        """Evaluates the elastic energy and the point forces
        Keyword Arguments:
        disp   -- array of distances
        pot    -- (default True) if true, returns potential energy
        forces -- (default False) if true, returns forces
        """
        if forces:
            dV = self.evaluateForce(disp)
            V = self.evaluateElasticEnergy(dV, disp)
        else:
            Fforce = self.evaluateKForce(disp)
            V = self.evaluateElasticEnergyKspace(Fforce, fftn(disp))
            dV = None
        return V, dV
if __name__ == '__main__':
    print(FFTElasticHalfSpace(512, 14.8))
    print(FFTElasticHalfSpace((512, 256), 14.8))
    print(FFTElasticHalfSpace(512, 14.8, 12.5))
    print(FFTElasticHalfSpace((512, 256), 14.8, 12.5))
    print(FFTElasticHalfSpace((512, 256), 14.8, (12.5, 28.3)))

    s_res = 512
    test_res = (s_res, s_res)
    hs = FFTElasticHalfSpace(test_res, 1, (12.5, 28.3))
    forces = np.zeros(test_res)
    forces[:s_res//2,:s_res//2] = 1

    import time
    start = time.perf_counter()
    disp = hs.evaluate_disp(forces)
    finish = time.perf_counter()
    print("Took {} seconds for a {}x{} grid".format(finish-start, *test_res))
    import matplotlib.pyplot as plt

    plt.contour(disp)
    plt.show()
