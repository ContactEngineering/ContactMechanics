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

class PeriodicFFTElasticHalfSpace(ElasticSubstrate):
    """ Uses the FFT to solve the displacements and stresses in an elastic
        Halfspace due to a given array of point forces. This halfspace implemen-
        tation cheats somewhat: since a net pressure would result in infinite
        displacement, the first term of the FFT is systematically dropped.
        The implementation follows the description in Stanley & Kato J. Tribol.
        119(3), 481-485 (Jul 01, 1997)
    """
    name = "periodic_fft_elastic_halfspace"
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
        self.steps = tuple(
            float(size)/res for size, res in zip(self.size, self.resolution))
        self.young = young

        self._computeFourierCoeffs()
        self._computeIFourierCoeffs()

    @property
    def dim(self, ):
        return self.__dim

    @property
    def computational_resolution(self, ):
        """
        usually, the resolution of the system is equal to the geometric
        resolution (of the surface). For example free boundary conditions,
        require the computational resolution to differ from the geometric one,
        see FreeFFTElasticHalfSpace.
        """
        return self.resolution

    def __repr__(self):
        dims = 'x', 'y', 'z'
        size_str = ', '.join('{}: {}({})'.format(dim, size, resolution) for
                             dim, size, resolution in zip(dims, self.size,
                                                          self.resolution))
        return ("{0.dim}-dimensional halfspace '{0.name}', size(resolution) in "
                "{1}, E' = {0.young}").format(self, size_str)

    def _computeFourierCoeffs(self):
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

    def _computeIFourierCoeffs(self):
        """Invert the weights w relating fft(displacement) to fft(pressure):
        """
        self.iweights = np.zeros_like(self.weights)
        self.iweights[self.weights != 0] = 1./self.weights[self.weights != 0]

    def evaluateDisp(self, forces):
        """ Computes the displacement due to a given force array
        Keyword Arguments:
        forces   -- a numpy array containing point forces (*not* pressures)
        """
        if forces.shape != self.computational_resolution:
            raise self.Error(
                ("force array has a different shape ({0}) than this halfspace's"
                 " resolution ({1})").format(
                     yforces.shape, self.computational_resolution))
        return ifftn(self.weights * fftn(-forces)).real/self.area_per_pt

    def evaluateForce(self, disp):
        """ Computes the force due to a given displacement array
        Keyword Arguments:
        disp   -- a numpy array containing point displacements
        """
        if disp.shape != self.computational_resolution:
            raise self.Error(
                ("force array has a different shape ({0}) than this halfspace's"
                 " resolution ({1})").format(
                     disp.shape, self.computational_resolution))
        return -ifftn(self.iweights*fftn(disp)).real*self.area_per_pt


    def evaluateKDisp(self, forces):
        """ Computes the K-space displacement due to a given force array
        Keyword Arguments:
        forces   -- a numpy array containing point forces (*not* pressures)
        """
        if forces.shape != self.computational_resolution:
            raise self.Error(
                ("force array has a different shape ({0}) than this halfspace's"
                 " resolution ({1})").format(
                     forces.shape, self.computational_resolution))
        return self.weights * fftn(forces)/self.area_per_pt

    def evaluateKForce(self, disp):
        """ Computes the K-space forces due to a given displacement array
        Keyword Arguments:
        disp   -- a numpy array containing point displacements
        """
        if disp.shape != self.computational_resolution:
            raise self.Error(
                ("force array has a different shape ({0}) than this halfspace's"
                 " resolution ({1})").format(
                     disp.shape, self.computational_resolution))
        return self.iweights*fftn(disp)*self.area_per_pt

    def evaluateElasticEnergy(self, forces, disp):
        return .5*np.dot(np.ravel(disp), np.ravel(-forces))

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


class FreeFFTElasticHalfSpace(PeriodicFFTElasticHalfSpace):
    """
    Uses the FFT to solve the displacements and stresses in an non-periodic
    elastic Halfspace due to a given array of point forces. Uses the Green's
    functions formulaiton of Johnson (1985, p. 54). The application of the FFT
    to a nonperiodic domain is explained in Hockney (1969, p. 178.)

    K. L. Johnson. (1985). Contact Mechanics. [Online]. Cambridge: Cambridge
    University Press. Available from: Cambridge Books Online
    <http://dx.doi.org/10.1017/CBO9781139171731> [Accessed 16 February 2015]

    R. W. HOCKNEY, "The potential calculation and some applications," Methods of
    Computational Physics, B. Adler, S. Fernback and M. Rotenberg (Eds.),
    Academic Press, New York, 1969, pp. 136-211.
    """
    name = "free_fft_elastic_halfspace"

    def __init__(self, resolution, young, size=2*np.pi, save_memory=False):
        """
        Keyword Arguments:
        resolution  -- Tuple containing number of points in spatial directions.
                       The length of the tuple determines the spatial dimension
                       of the problem. Warning: internally, the free boundary
                       conditions require the system so store a system of
                       2*resolution.x by 2*resolution.y. Keep in mind that if
                       your surface is nx by ny, the forces and displacements
                       will still be 2nx by 2ny.
        young       -- Equiv. Young's modulus E'
                       1/E' = (i-ν_1**2)/E'_1 + (i-ν_2**2)/E'_2
        size        -- (default 2π) domain size. For multidimensional problems,
                       a tuple can be provided to specify the lenths per
                       dimension. If the tuple has less entries than dimensions,
                       the last value in repeated.
        save_memory -- (default False) if this flag is set, a less memory-hungry
                       method to compute the fourier coefficients is used. The
                       less memory-hungry method is slower.
        """
        self.save_memory = save_memory
        super().__init__(resolution, young, size)
        self._comp_resolution = tuple((2*r for r in self.resolution))

    @property
    def computational_resolution(self, ):
        """
        usually, the resolution of the system is equal to the geometric
        resolution (of the surface). For example free boundary conditions,
        require the computational resolution to differ from the geometric one,
        see FreeFFTElasticHalfSpace.
        """
        return self._comp_resolution

    def _computeFourierCoeffs(self):
        """ Delegates the computation of the weights w relating
            fft(displacement) to fft(pressure):
            fft(u) = w*fft(p), Johnson, p. 54, and Hockney, p. 178
        """
        if self.save_memory:
            return self._computeFourierCoeffs3()
        return self._computeFourierCoeffs2()
    def _computeFourierCoeffs0(self):
        """Compute the weights w relating fft(displacement) to fft(pressure):
           fft(u) = w*fft(p), Johnson, p. 54, and Hockney, p. 178
           This version has no optimizations an should not be used. it remains
           here as a reference for the unit testing because of it's simple
           uggability
        """
        facts = np.zeros(tuple((res*2 for res in self.resolution)))
        a = self.steps[0]*.5
        if self.dim == 1:
            pass
        else:
            b = self.steps[1]*.5
            nx, ny = (res*2-1 for res in self.resolution)
            for i in range (self.resolution[0]):
                x = self.steps[0]*i
                for j in range (self.resolution[1]):
                    y = self.steps[1]*j
                    facts[i, j] = facts[nx-i, j] = facts[i, ny-j] = \
                      facts[nx-i, ny-j] = 1/(np.pi*self.young)*\
                      ( (x+a)*np.log(((y+b)+np.sqrt((y+b)*(y+b)+(x+a)*(x+a)))/
                                     ((y-b)+np.sqrt((y-b)*(y-b)+(x+a)*(x+a))))
                       +(y+b)*np.log(((x+a)+np.sqrt((y+b)*(y+b)+(x+a)*(x+a)))/
                                     ((x-a)+np.sqrt((y+b)*(y+b)+(x-a)*(x-a))))
                       +(x-a)*np.log(((y-b)+np.sqrt((y-b)*(y-b)+(x-a)*(x-a)))/
                                     ((y+b)+np.sqrt((y+b)*(y+b)+(x-a)*(x-a))))
                       +(y-b)*np.log(((x-a)+np.sqrt((y-b)*(y-b)+(x-a)*(x-a)))/
                                     ((x+a)+np.sqrt((y-b)*(y-b)+(x+a)*(x+a)))))
        self.weights = fftn(facts)
        return self.weights, facts

    def _computeFourierCoeffs1(self):
        """Compute the weights w relating fft(displacement) to fft(pressure):
           fft(u) = w*fft(p), Johnson, p. 54, and Hockney, p. 178
           this Version has worthless python optimizations and should not be
           used. merely here for historical reasons.
        """
        facts = np.zeros(tuple((res*2 for res in self.resolution)))
        a = self.steps[0]*.5
        if self.dim == 1:
            pass
        else:
            b = self.steps[1]*.5
            nx, ny = (res*2-1 for res in self.resolution)
            for i in range (self.resolution[0]):
                x = self.steps[0]*i
                xpa = x+a
                xma = x-a
                xpa2 = xpa*xpa
                xma2 = xma*xma
                for j in range (self.resolution[1]):
                    y = self.steps[1]*j
                    ypb = y+b
                    ymb = y-b
                    ypb2 = ypb*ypb
                    ymb2 = ymb*ymb
                    facts[i, j] = facts[nx-i, j] = facts[i, ny-j] = \
                      facts[nx-i, ny-j] = 1/(np.pi*self.young)*\
                      ( xpa*np.log((ypb+np.sqrt(ypb2+xpa2))/
                                   (ymb+np.sqrt(ymb2+xpa2)))
                       +ypb*np.log((xpa+np.sqrt(ypb2+xpa2))/
                                   (xma+np.sqrt(ypb2+xma2)))
                       +xma*np.log((ymb+np.sqrt(ymb2+xma2))/
                                   (ypb+np.sqrt(ypb2+xma2)))
                       +ymb*np.log((xma+np.sqrt(ymb2+xma2))/
                                   (xpa+np.sqrt(ymb2+xpa2))))
        self.weights = fftn(facts)
        return self.weights, facts

    def _computeFourierCoeffs2(self):
        """Compute the weights w relating fft(displacement) to fft(pressure):
           fft(u) = w*fft(p), Johnson, p. 54, and Hockney, p. 178

           This is the fastest version, about 2 orders faster than the python
           versions, however a bit memory-hungry, this version is used by default
        """
        facts = np.zeros(tuple((res*2 for res in self.resolution)))
        a = self.steps[0]*.5
        if self.dim == 1:
            pass
        else:
            b = self.steps[1]*.5
            xp = (np.arange(self.resolution[0])*self.steps[0]+a).reshape((-1, 1))
            xm = (np.arange(self.resolution[0])*self.steps[0]-a).reshape((-1, 1))
            xp2 = xp*xp
            xm2 = xm*xm

            yp = np.arange(self.resolution[1])*self.steps[1]+b
            ym = np.arange(self.resolution[1])*self.steps[1]-b
            yp2 = yp*yp
            ym2 = ym*ym
            sqrt_yp_xp = np.sqrt(np.zeros(self.resolution) + yp2 + xp2)
            sqrt_ym_xp = np.sqrt(np.zeros(self.resolution) + ym2 + xp2)
            sqrt_yp_xm = np.sqrt(np.zeros(self.resolution) + yp2 + xm2)
            sqrt_ym_xm = np.sqrt(np.zeros(self.resolution) + ym2 + xm2)
            facts[:self.resolution[0], -1:self.resolution[1]-1:-1] = \
              facts[-1:self.resolution[0]-1:-1, :self.resolution[1]] = \
              facts[-1:self.resolution[0]-1:-1, -1:self.resolution[1]-1:-1] = \
              facts[:self.resolution[0], :self.resolution[1]] = 1/(np.pi*self.young)*\
              ( xp*np.log((yp+sqrt_yp_xp)/
                          (ym+sqrt_ym_xp))
               +yp*np.log((xp+sqrt_yp_xp)/
                          (xm+sqrt_yp_xm))
               +xm*np.log((ym+sqrt_ym_xm)/
                          (yp+sqrt_yp_xm))
               +ym*np.log((xm+sqrt_ym_xm)/
                          (xp+sqrt_ym_xp)))
        self.weights = fftn(facts)
        return self.weights, facts

    def _computeFourierCoeffs3(self):
        """Compute the weights w relating fft(displacement) to fft(pressure):
           fft(u) = w*fft(p), Johnson, p. 54, and Hockney, p. 178

           This version is less memory-hungry than the default version, but also
           only about half as fast. Use if memory is a concern.
        """
        facts = np.zeros(tuple((res*2 for res in self.resolution)))
        a = self.steps[0]*.5
        if self.dim == 1:
            pass
        else:
            b = self.steps[1]*.5
            xp = (np.arange(self.resolution[0])*self.steps[0]+a).reshape((-1, 1))
            xm = (np.arange(self.resolution[0])*self.steps[0]-a).reshape((-1, 1))
            xp2 = xp*xp
            xm2 = xm*xm

            yp = np.arange(self.resolution[1])*self.steps[1]+b
            ym = np.arange(self.resolution[1])*self.steps[1]-b
            yp2 = yp*yp
            ym2 = ym*ym
            facts[:self.resolution[0], -1:self.resolution[1]-1:-1] = \
              facts[-1:self.resolution[0]-1:-1, :self.resolution[1]] = \
              facts[-1:self.resolution[0]-1:-1, -1:self.resolution[1]-1:-1] = \
              facts[:self.resolution[0], :self.resolution[1]] = 1/(np.pi*self.young)*\
              ( xp*np.log((yp+np.sqrt(np.zeros(self.resolution) + yp2 + xp2))/
                          (ym+np.sqrt(np.zeros(self.resolution) + ym2 + xp2)))
               +yp*np.log((xp+np.sqrt(np.zeros(self.resolution) + yp2 + xp2))/
                          (xm+np.sqrt(np.zeros(self.resolution) + yp2 + xm2)))
               +xm*np.log((ym+np.sqrt(np.zeros(self.resolution) + ym2 + xm2))/
                          (yp+np.sqrt(np.zeros(self.resolution) + yp2 + xm2)))
               +ym*np.log((xm+np.sqrt(np.zeros(self.resolution) + ym2 + xm2))/
                          (xp+np.sqrt(np.zeros(self.resolution) + ym2 + xp2))))
        self.weights = fftn(facts)
        return self.weights, facts

if __name__ == '__main__':
    print(PeriodicFFTElasticHalfSpace(512, 14.8))
    print(PeriodicFFTElasticHalfSpace((512, 256), 14.8))
    print(PeriodicFFTElasticHalfSpace(512, 14.8, 12.5))
    print(PeriodicFFTElasticHalfSpace((512, 256), 14.8, 12.5))
    print(PeriodicFFTElasticHalfSpace((512, 256), 14.8, (12.5, 28.3)))

    s_res = 512
    test_res = (s_res, s_res)
    hs = PeriodicFFTElasticHalfSpace(test_res, 1, (12.5, 28.3))
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
