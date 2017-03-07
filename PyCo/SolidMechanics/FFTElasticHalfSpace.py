#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   FFTElasticHalfSpace.py

@author Till Junge <till.junge@kit.edu>

@date   26 Jan 2015

@brief  Implement the FFT-based elasticity solver of pycontact

@section LICENCE

 Copyright (C) 2015 Till Junge

PyCo is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation, either version 3, or (at
your option) any later version.

PyCo is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with GNU Emacs; see the file COPYING. If not, write to the
Free Software Foundation, Inc., 59 Temple Place - Suite 330,
Boston, MA 02111-1307, USA.
"""

from collections import namedtuple
import numpy as np
from ..Tools.fftext import rfftn, irfftn

from .Substrates import ElasticSubstrate


class PeriodicFFTElasticHalfSpace(ElasticSubstrate):
    """ Uses the FFT to solve the displacements and stresses in an elastic
        Halfspace due to a given array of point forces. This halfspace
        implementation cheats somewhat: since a net pressure would result in
        infinite displacement, the first term of the FFT is systematically
        dropped.
        The implementation follows the description in Stanley & Kato J. Tribol.
        119(3), 481-485 (Jul 01, 1997)
    """
    # pylint: disable=too-many-instance-attributes

    name = "periodic_fft_elastic_halfspace"
    _periodic = True

    def __init__(self, resolution, young, size=2*np.pi, stiffness_q0=0.0,
                 superclass=True):
        """
        Keyword Arguments:
        resolution   -- Tuple containing number of points in spatial directions.
                        The length of the tuple determines the spatial dimension
                        of the problem.
        young        -- Equiv. Young's modulus E'
                        1/E' = (i-ν_1**2)/E'_1 + (i-ν_2**2)/E'_2
        size         -- (default 2π) domain size. For multidimensional problems,
                        a tuple can be provided to specify the lenths per
                        dimension. If the tuple has less entries than dimensions,
                        the last value in repeated.
        stiffness_q0 -- Substrate stiffness at the Gamma-point (wavevector q=0).
                        If None, this is taken equal to the lowest nonvanishing
                        stiffness.
        superclass   -- (default True) client software never uses this. Only
                        inheriting subclasses use this.
        """
        super().__init__()
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
        try:
            self.steps = tuple(
                float(size)/res for size, res in
                zip(self.size, self.resolution))
        except ZeroDivisionError as err:
            raise ZeroDivisionError(
                ("{}, when trying to handle "
                 "    self.steps = tuple("
                 "        float(size)/res for size, res in"
                 "        zip(self.size, self.resolution))"
                 "Parameters: self.size = {}, self.resolution = {}"
                 "").format(err, self.size, self.resolution))
        self.young = young
        self.stiffness_q0 = stiffness_q0
        if superclass:
            self._compute_fourier_coeffs()
            self._compute_i_fourier_coeffs()

    @property
    def dim(self, ):
        "return the substrate's physical dimension"
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
        return ("{0.dim}-dimensional halfspace '{0.name}', size(resolution) in"
                " {1}, E' = {0.young}").format(self, size_str)

    def _compute_fourier_coeffs(self):
        """
        Compute the weights w relating fft(displacement) to fft(pressure):
        fft(u) = w*fft(p), see (6) Stanley & Kato J. Tribol. 119(3), 481-485
        (Jul 01, 1997).
        WARNING: the paper is dimensionally *incorrect*. see for the correct
        1D formulation: Section 13.2 in
            K. L. Johnson. (1985). Contact Mechanics. [Online]. Cambridge:
            Cambridge  University Press. Available from: Cambridge Books Online
            <http://dx.doi.org/10.1017/CBO9781139171731> [Accessed 16 February
            2015]
        for correct 2D formulation: Appendix 1, eq A.2 in
            Johnson, Greenwood and Higginson, "The Contact of Elastic Regular
            Wavy surfaces", Int. J. Mech. Sci. Vol. 27 No. 6, pp. 383-396, 1985
            <http://dx.doi.org/10.1016/0020-7403(85)90029-3> [Accessed 18 March
            2015]
        """
        if self.dim == 1:
            facts = np.zeros(self.resolution)
            for index in range(2, self.resolution[0]//2+2):
                facts[-index+1] = facts[index - 1] = \
                    self.size[0]/(self.young*index*np.pi)
            # real to complex fft allows saving some time and memory
            nb_cols = facts.shape[-1]//2+1
            self.weights = facts[..., :nb_cols]

        elif self.dim == 2:
            nx, ny = self.resolution
            sx, sy = self.size
            qx = np.arange(nx, dtype=np.float64)
            qx = np.where(qx <= nx//2, qx/sx, (nx-qx)/sx)
            qy = np.arange(ny, dtype=np.float64)
            qy = np.where(qy <= ny//2, qy/sy, (ny-qy)/sy)
            facts = np.pi*self.young*np.sqrt((qx*qx).reshape(-1, 1) +
                                             (qy*qy).reshape(1, -1))
            if self.stiffness_q0 is None:
                facts[0, 0] = (facts[1, 0].real + facts[0, 1].real)/2
            elif self.stiffness_q0 == 0.0:
                facts[0, 0] = 1.0
            else:
                facts[0, 0] = self.stiffness_q0
            # real to complex fft allows saving some time and memory
            nb_cols = facts.shape[-1]//2+1
            self.weights = 1/facts[..., :nb_cols]
            if self.stiffness_q0 == 0.0:
                self.weights[0, 0] = 0.0

    def _compute_i_fourier_coeffs(self):
        """Invert the weights w relating fft(displacement) to fft(pressure):
        """
        self.iweights = np.zeros_like(self.weights)
        self.iweights[self.weights != 0] = 1./self.weights[self.weights != 0]

    def evaluate_disp(self, forces):
        """ Computes the displacement due to a given force array
        Keyword Arguments:
        forces   -- a numpy array containing point forces (*not* pressures)
        """
        if forces.shape != self.computational_resolution:
            raise self.Error(
                ("force array has a different shape ({0}) than this halfspace'"
                 "s resolution ({1})").format(
                     forces.shape, self.computational_resolution))  # nopep8
        return irfftn(self.weights * rfftn(-forces), s=self.computational_resolution).real/self.area_per_pt

    def evaluate_force(self, disp):
        """ Computes the force (*not* pressures) due to a given displacement array
        Keyword Arguments:
        disp   -- a numpy array containing point displacements
        """
        if disp.shape != self.computational_resolution:
            raise self.Error(
                ("force array has a different shape ({0}) than this halfspace'"
                 "s resolution ({1})").format(
                     disp.shape, self.computational_resolution))  # nopep8
        return -irfftn(self.iweights*rfftn(disp), s=self.computational_resolution).real*self.area_per_pt

    def evaluate_k_disp(self, forces):
        """ Computes the K-space displacement due to a given force array
        Keyword Arguments:
        forces   -- a numpy array containing point forces (*not* pressures)
        """
        if forces.shape != self.computational_resolution:
            raise self.Error(
                ("force array has a different shape ({0}) than this halfspace'"
                 "s resolution ({1})").format(
                     forces.shape, self.computational_resolution))  # nopep8
        return self.weights * rfftn(-forces)/self.area_per_pt

    def evaluate_k_force(self, disp):
        """ Computes the K-space forces (*not* pressures) due to a given displacement array
        Keyword Arguments:
        disp   -- a numpy array containing point displacements
        """
        if disp.shape != self.computational_resolution:
            raise self.Error(
                ("force array has a different shape ({0}) than this halfspace'"
                 "s resolution ({1})").format(
                     disp.shape, self.computational_resolution))  # nopep8
        return -self.iweights*rfftn(disp)*self.area_per_pt

    def evaluate_elastic_energy(self, forces, disp):
        """
        computes and returns the elastic energy due to forces and displacements
        Arguments:
        forces -- array of forces
        disp   -- array of displacements
        """
        # pylint: disable=no-self-use
        return .5*np.dot(np.ravel(disp), np.ravel(-forces))

    def evaluate_elastic_energy_k_space(self, kforces, kdisp):
        """
        computes and returns the elastic energy due to forces and displacement
        in Fourier space
        Arguments:
        forces -- array of forces
        disp   -- array of displacements
        """
        # pylint: disable=no-self-use
        # using vdot instead of dot because of conjugate
        # The 2nd term at the end comes from the fact that we use a reduced
        # rfft transform
        return .5*(np.vdot(kdisp, -kforces).real +
                   np.vdot(kdisp[..., :-1], -kforces[..., :-1]).real)/self.nb_pts

    def evaluate(self, disp, pot=True, forces=False):
        """Evaluates the elastic energy and the point forces
        Keyword Arguments:
        disp   -- array of distances
        pot    -- (default True) if true, returns potential energy
        forces -- (default False) if true, returns forces
        """
        force = potential = None
        if forces:
            force = self.evaluate_force(disp)
            if pot:
                potential = self.evaluate_elastic_energy(force, disp)
        elif pot:
            kforce = self.evaluate_k_force(disp)
            potential = self.evaluate_elastic_energy_k_space(
                kforce, rfftn(disp))
        return potential, force


class FreeFFTElasticHalfSpace(PeriodicFFTElasticHalfSpace):
    """
    Uses the FFT to solve the displacements and stresses in an non-periodic
    elastic Halfspace due to a given array of point forces. Uses the Green's
    functions formulaiton of Johnson (1985, p. 54). The application of the FFT
    to a nonperiodic domain is explained in Hockney (1969, p. 178.)

    K. L. Johnson. (1985). Contact Mechanics. [Online]. Cambridge: Cambridge
    University Press. Available from: Cambridge Books Online
    <http://dx.doi.org/10.1017/CBO9781139171731> [Accessed 16 February 2015]

    R. W. HOCKNEY, "The potential calculation and some applications," Methods
    of Computational Physics, B. Adler, S. Fernback and M. Rotenberg (Eds.),
    Academic Press, New York, 1969, pp. 136-211.
    """
    name = "free_fft_elastic_halfspace"
    _periodic = False

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
                       dimension. If the tuple has less entries than
                       dimensions, the last value in repeated.
        """
        super().__init__(resolution, young, size, superclass=False)
        self._compute_fourier_coeffs()
        self._compute_i_fourier_coeffs()
        self._comp_resolution = tuple((2*r for r in self.resolution))

    def spawn_child(self, resolution):
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

    def _compute_fourier_coeffs2(self):
        """Compute the weights w relating fft(displacement) to fft(pressure):
           fft(u) = w*fft(p), Johnson, p. 54, and Hockney, p. 178

           Now Deprecated
           This is the fastest version, about 2 orders faster than the python
           versions, however a bit memory-hungry, this version used to be
           default, but turns out to have no significant advantage over the
           matscipy implementation
        """
        # pylint: disable=too-many-locals
        if self.dim == 1:
            pass
        else:
            x_grid = np.arange(self.resolution[0]*2)
            x_grid = np.where(x_grid <= self.resolution[0], x_grid,
                              x_grid-self.resolution[0]*2) * self.steps[0]
            x_grid.shape = (-1, 1)
            y_grid = np.arange(self.resolution[1]*2)
            y_grid = np.where(y_grid <= self.resolution[1], y_grid,
                              y_grid-self.resolution[1]*2) * self.steps[1]
            y_grid.shape = (1, -1)
            x_p = (x_grid+self.steps[0]*.5).reshape((-1, 1))
            x_m = (x_grid-self.steps[0]*.5).reshape((-1, 1))
            xp2 = x_p*x_p
            xm2 = x_m*x_m

            y_p = y_grid+self.steps[1]*.5
            y_m = y_grid-self.steps[1]*.5
            yp2 = y_p*y_p
            ym2 = y_m*y_m
            sqrt_yp_xp = np.sqrt(yp2 + xp2)
            sqrt_ym_xp = np.sqrt(ym2 + xp2)
            sqrt_yp_xm = np.sqrt(yp2 + xm2)
            sqrt_ym_xm = np.sqrt(ym2 + xm2)
            facts = 1/(np.pi * self.young) * (
                x_p * np.log((y_p+sqrt_yp_xp) /
                             (y_m+sqrt_ym_xp)) +
                y_p * np.log((x_p+sqrt_yp_xp) /
                             (x_m+sqrt_yp_xm)) +
                x_m * np.log((y_m+sqrt_ym_xm) /
                             (y_p+sqrt_yp_xm)) +
                y_m * np.log((x_m+sqrt_ym_xm) /
                             (x_p+sqrt_ym_xp)))
        return rfftn(facts), facts

    def _compute_fourier_coeffs(self):
        """Compute the weights w relating fft(displacement) to fft(pressure):
           fft(u) = w*fft(p), Johnson, p. 54, and Hockney, p. 178

           This version is less is copied from matscipy, use if memory is a
           concern
        """
        # pylint: disable=invalid-name
        facts = np.zeros(tuple((res*2 for res in self.resolution)))
        a = self.steps[0]*.5
        if self.dim == 1:
            pass
        else:
            b = self.steps[1]*.5
            x_s = np.arange(self.resolution[0]*2)
            x_s = np.where(x_s <= self.resolution[0], x_s,
                           x_s-self.resolution[0] * 2) * self.steps[0]
            x_s.shape = (-1, 1)
            y_s = np.arange(self.resolution[1]*2)
            y_s = np.where(y_s <= self.resolution[1], y_s,
                           y_s-self.resolution[1]*2) * self.steps[1]
            y_s.shape = (1, -1)
            facts = 1/(np.pi*self.young) * (
                (x_s+a)*np.log(((y_s+b)+np.sqrt((y_s+b)*(y_s+b) +
                                                (x_s+a)*(x_s+a))) /
                               ((y_s-b)+np.sqrt((y_s-b)*(y_s-b) +
                                                (x_s+a)*(x_s+a)))) +
                (y_s+b)*np.log(((x_s+a)+np.sqrt((y_s+b)*(y_s+b) +
                                                (x_s+a)*(x_s+a))) /
                               ((x_s-a)+np.sqrt((y_s+b)*(y_s+b) +
                                                (x_s-a)*(x_s-a)))) +
                (x_s-a)*np.log(((y_s-b)+np.sqrt((y_s-b)*(y_s-b) +
                                                (x_s-a)*(x_s-a))) /
                               ((y_s+b)+np.sqrt((y_s+b)*(y_s+b) +
                                                (x_s-a)*(x_s-a)))) +
                (y_s-b)*np.log(((x_s-a)+np.sqrt((y_s-b)*(y_s-b) +
                                                (x_s-a)*(x_s-a))) /
                               ((x_s+a)+np.sqrt((y_s-b)*(y_s-b) +
                                                (x_s+a)*(x_s+a)))))
        self.weights = rfftn(facts)
        return self.weights, facts

    def evaluate_disp(self, forces):
        """ Computes the displacement due to a given force array
        Keyword Arguments:
        forces   -- a numpy array containing point forces (*not* pressures)
        """
        if forces.shape != self.computational_resolution:
            # Automatically pad forces if force array is half of computational
            # resolution
            if np.any(np.array(forces.shape) !=
                      np.array(self.computational_resolution)/2):
                raise self.Error("force array has a different shape ({0}) "
                                 "than this halfspace's resolution ({1}) or "
                                 "half of it".format(forces.shape,
                                                     self.computational_resolution))
            padded_forces = np.zeros(self.computational_resolution)
            s = [slice(0, forces.shape[i])
                 for i in range(len(forces.shape))]
            padded_forces[s] = forces
            return super().evaluate_disp(padded_forces)[s]
        else:
            return super().evaluate_disp(forces)

# convenient container for storing correspondences betwees small and large
# system
BndSet = namedtuple('BndSet', ('large', 'small'))
