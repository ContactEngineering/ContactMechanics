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
from collections import namedtuple

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
    def __init__(self, resolution, young, size=2*np.pi, superclass=True):
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
        superclass -- (default True) client software never uses this. Only
                      inheriting subclasses use this.
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
        if superclass:
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
        ## using vdot instead of dot because of conjugate
        return .5*np.vdot(Kdisp, Kforces).real/self.nb_pts

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

    ### This class should not be used directly, as it uses a fixed size
    ### computational domain instead of determining it dynamically to safe
    ### resources. Instead, use the proxy-class FastFreeFFTElasticHalfSpace, which
    ### has the same interface.
    """
    name = "free_fft_elastic_halfspace"

    def __init__(self, resolution, young, size=2*np.pi):
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
        """
        super().__init__(resolution, young, size, superclass=False)
        self._computeFourierCoeffs()
        self._computeIFourierCoeffs()
        self._comp_resolution = tuple((2*r for r in self.resolution))

    def spawnChild(self, resolution):
        """
        returns an instance with same physical properties with a smaller
        computational grid
        """
        size = tuple((resolution[i]/float(self.resolution[i])*self.size[i] for
                      i in range(self.dim)))
        return type(self)(resolution, self.young, size)

    @property
    def computational_resolution(self, ):
        """
        usually, the resolution of the system is equal to the geometric
        resolution (of the surface). For example free boundary conditions,
        require the computational resolution to differ from the geometric one,
        see FreeFFTElasticHalfSpace.
        """
        return self._comp_resolution

    def _computeFourierCoeffs2(self):
        """Compute the weights w relating fft(displacement) to fft(pressure):
           fft(u) = w*fft(p), Johnson, p. 54, and Hockney, p. 178

           Now Deprecated
           This is the fastest version, about 2 orders faster than the python
           versions, however a bit memory-hungry, this version used to be
           default, but turns out to have no significant advantage over the
           matscipy implementation
        """
        facts = np.zeros(tuple((res*2 for res in self.resolution)))
        a = self.steps[0]*.5
        if self.dim == 1:
            pass
        else:
            b = self.steps[1]*.5
            x = np.arange(self.resolution[0]*2)
            x = np.where(x <= self.resolution[0], x, x-self.resolution[0]*2) * self.steps[0]
            x.shape = (-1,1)
            y = np.arange(self.resolution[1]*2)
            y = np.where(y <= self.resolution[1], y, y-self.resolution[1]*2) * self.steps[1]
            y.shape = (1,-1)
            xp = (x+a).reshape((-1, 1))
            xm = (x-a).reshape((-1, 1))
            xp2 = xp*xp
            xm2 = xm*xm

            yp = y+b
            ym = y-b
            yp2 = yp*yp
            ym2 = ym*ym
            sqrt_yp_xp = np.sqrt(yp2 + xp2)
            sqrt_ym_xp = np.sqrt(ym2 + xp2)
            sqrt_yp_xm = np.sqrt(yp2 + xm2)
            sqrt_ym_xm = np.sqrt(ym2 + xm2)
            facts = 1/(np.pi*self.young)*\
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

    def _computeFourierCoeffs(self):
        """Compute the weights w relating fft(displacement) to fft(pressure):
           fft(u) = w*fft(p), Johnson, p. 54, and Hockney, p. 178

           This version is less is copied from matscipy, use if memory is a concern
        """
        facts = np.zeros(tuple((res*2 for res in self.resolution)))
        a = self.steps[0]*.5
        if self.dim == 1:
            pass
        else:
            b = self.steps[1]*.5
            x = np.arange(self.resolution[0]*2)
            x = np.where(x <= self.resolution[0], x, x-self.resolution[0]*2)* self.steps[0]
            x.shape = (-1,1)
            y = np.arange(self.resolution[1]*2)
            y = np.where(y <= self.resolution[1], y, y-self.resolution[1]*2)* self.steps[1]
            y.shape = (1,-1)
            facts = 1/(np.pi*self.young)*\
              ( (x+a)*np.log( ( (y+b)+np.sqrt((y+b)*(y+b)+(x+a)*(x+a)) )/
                              ( (y-b)+np.sqrt((y-b)*(y-b)+(x+a)*(x+a)) ) )+
                (y+b)*np.log( ( (x+a)+np.sqrt((y+b)*(y+b)+(x+a)*(x+a)) ) /
                              ( (x-a)+np.sqrt((y+b)*(y+b)+(x-a)*(x-a)) ) )+
                (x-a)*np.log( ( (y-b)+np.sqrt((y-b)*(y-b)+(x-a)*(x-a)) ) /
                              ( (y+b)+np.sqrt((y+b)*(y+b)+(x-a)*(x-a)) ) )+
                (y-b)*np.log( ( (x-a)+np.sqrt((y-b)*(y-b)+(x-a)*(x-a)) ) /
                              ( (x+a)+np.sqrt((y-b)*(y-b)+(x+a)*(x+a)) ) ) )
        self.weights = fftn(facts)
        return self.weights, facts


## convenient container for storing correspondences betwees small and large system
BndSet = namedtuple('BndSet', ('large', 'small'))

## class FastFreeFFTElasticHalfSpace(PeriodicFFTElasticHalfSpace):
##     """
##     Uses the FFT to solve the displacements and stresses in an non-periodic
##     elastic Halfspace due to a given array of point forces. Uses the Green's
##     functions formulaiton of Johnson (1985, p. 54). The application of the FFT
##     to a nonperiodic domain is explained in Hockney (1969, p. 178.)
## 
##     K. L. Johnson. (1985). Contact Mechanics. [Online]. Cambridge: Cambridge
##     University Press. Available from: Cambridge Books Online
##     <http://dx.doi.org/10.1017/CBO9781139171731> [Accessed 16 February 2015]
## 
##     R. W. HOCKNEY, "The potential calculation and some applications," Methods of
##     Computational Physics, B. Adler, S. Fernback and M. Rotenberg (Eds.),
##     Academic Press, New York, 1969, pp. 136-211.
##     """
##     name = "free_fft_elastic_halfspace"
## 
##     def __init__(self, resolution, young, size=2*np.pi, buffer_zone=16):
##         """
##         Keyword Arguments:
##         resolution  -- Tuple containing number of points in spatial directions.
##                        The length of the tuple determines the spatial dimension
##                        of the problem. Warning: internally, the free boundary
##                        conditions require the system so store a system of
##                        2*resolution.x by 2*resolution.y. Keep in mind that if
##                        your surface is nx by ny, the forces and displacements
##                        will still be 2nx by 2ny.
##         young       -- Equiv. Young's modulus E'
##                        1/E' = (i-ν_1**2)/E'_1 + (i-ν_2**2)/E'_2
##         size        -- (default 2π) domain size. For multidimensional problems,
##                        a tuple can be provided to specify the lenths per
##                        dimension. If the tuple has less entries than dimensions,
##                        the last value in repeated.
##         buffer-zone -- (default 16) number of pixels around the contact area
##                        bounding box to be included
##         """
##         super().__init__(resolution, young, size)
##         self._comp_resolution = tuple((2*r for r in self.resolution))
## 
##         ## this is where the nested FreeFFTElasticHalfSpace will be stored
##         self.babushka = None
##         self.offset = tuple((0 for _ in self.resolution))
##         if self.dim != 2:
##             raise NotImplementedError(
##                 "FastFreeFFTElasticHalfSpace is currently only implemented for "
##                 "2-dimensional problems")
##         self.computeBabushkaBounds()
## 
##     @property
##     def needInit(self):
##         return True
## 
##    ## def init(self, system):
## 
##     def _computeFourierCoeffs(self): raise NotImplementedError
##     def _computeIFourierCoeffs(self): raise NotImplementedError
## 
##     def computeBabushkaBounds(self):
##         def boundary_generator():
##           sm_res = self.babushka.resolution
##           lg_res = self.resolution
##           for i in (0,1):
##             for j in (0,1):
##               sm_slice = tuple(slice(i*sm_res[0], (i+1)*sm_res[0]),
##                                slice(j*sm_res[1], (j+1)*sm_res[1]))
##               lg_slice = tuple(
##                   slice(i*lg_res[0]+self.offset[0], (i+1)*lg_res[0]+self.offset[0]),
##                   slice(j*lg_res[1]+self.offset[1], (j+1)*lg_res[1]+self.offset[1]))
##               yield(BndSet(large=lg_slice, small=sm_slice))
##         self.bounds = tuple((bnd for bnd in boundary_generator()))
## 
## 
## 
## 
##     def _getBabushkaArray(self, full_array, babushka_array=None):
##         if babushka_array is None:
##             babushka_array = np.zeros(self.babushka.computational_resolution)
##         for bnd in self.bounds:
##             babushka_array[bnd.small] = full_array[bnd.large]
##         return babushka_array
## 
##     def _getFullArray(self, babushka_array, full_array=None):
##         if full_array is None:
##             full_array = np.zeros(self.babushka.computational_resolution)
##         for bnd in self.bounds:
##             full_array[bnd.small] = babushka_array[bnd.large]
##         return full_array
## 
## 
##     def evaluateForce(self, disp):
##         return self._getFullArray(
##             self.babushka.evaluateForce(self._getBabushkaArray(disp)))
## 
##     def evaluateDisp(self, force):
##         return self._getFullArray(
##             self.babushka.evaluateDisp(self._getBabushkaArray(force)))
## 
##     def evaluateKForce(self, disp):
##         return self.babushka.evaluateKForce(self._getBabushkaArray(disp))
## 
##     def evaluateKDisp(self, forces):
##         return self.babushka.evaluateKDisp(self._getBabushkaArray(force))
##     
