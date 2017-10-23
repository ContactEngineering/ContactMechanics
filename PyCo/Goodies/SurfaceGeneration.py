#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   SurfaceGeneration.py

@author Till Junge <till.junge@kit.edu>

@date   18 Jun 2015

@brief  Helper functions for the generation of random fractal surfaces

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

import numpy as np
import scipy.stats as stats
from ..Surface import NumpySurface
from ..Tools.common import compute_wavevectors, ifftn, fftn
from .SurfaceAnalysis import CharacterisePeriodicSurface


class RandomSurfaceExact(object):
    """ Metasurface with exact power spectrum"""
    Error = Exception

    def __init__(self, resolution, size, hurst, rms_height=None,
                 rms_slope=None, seed=None, lambda_min=None, lambda_max=None):
        """
        Generates a surface with an exact power spectrum (deterministic
        amplitude)
        Keyword Arguments:
        resolution -- Tuple containing number of points in spatial directions.
                      The length of the tuple determines the spatial dimension
                      of the problem (for the time being, only 1D or square 2D)
        size       -- domain size. For multidimensional problems,
                      a tuple can be provided to specify the length per
                      dimension. If the tuple has less entries than dimensions,
                      the last value in repeated.
        hurst      -- Hurst exponent
        rms_height -- root mean square asperity height
        rms_slope  -- root mean square slope of surface
        seed       -- (default hash(None)) for repeatability, the random number
                      generator is seeded previous to outputting the generated
                      surface
        lambda_min -- (default None) min wavelength to consider when scaling
                      power spectral density
        lambda_max -- (default None) max wavelength to consider when scaling
                      power spectral density
        """
        if seed is not None:
            np.random.seed(hash(seed))
        if not hasattr(resolution, "__iter__"):
            resolution = (resolution, )
        if not hasattr(size, "__iter__"):
            size = (size, )

        self.dim = len(resolution)
        if self.dim not in (1, 2):
            raise self.Error(
                ("Dimension of this problem is {}. Only 1 and 2-dimensional "
                 "problems are supported").format(self.dim))
        self.resolution = resolution
        tmpsize = list()
        for i in range(self.dim):
            tmpsize.append(size[min(i, len(size)-1)])
        self.size = tuple(tmpsize)

        self.hurst = hurst

        if rms_height is None and rms_slope is None:
            raise self.Error('Please specify either rms height or rms slope.')
        if rms_height is not None and rms_slope is not None:
            raise self.Error('Please specify either rms height or rms slope, '
                             'not both.')

        self.rms_height = rms_height
        self.rms_slope = rms_slope
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        if lambda_max is not None:
            self.q_min = 2*np.pi/lambda_max
        else:
            self.q_min = 2*np.pi*max(1/self.size[0], 1/self.size[1])

        max_pixelsize = max(
            (siz/res for siz, res in zip(self.size, self.resolution)))
        self.q_max = np.pi/max_pixelsize

        self.prefactor = self.compute_prefactor()

        self.q = compute_wavevectors(  # pylint: disable=invalid-name
            self.resolution, self.size, self.dim)
        self.coeffs = self.generate_phases()
        self.generate_amplitudes()
        self.distribution = self.amplitude_distribution()
        self.active_coeffs = None

    def get_negative_frequency_iterator(self):
        " frequency complement"
        def iterator():  # pylint: disable=missing-docstring
            for i in range(self.resolution[0]):
                for j in range(self.resolution[1]//2+1):
                    yield (i, j), (-i, -j)
        return iterator()

    def amplitude_distribution(self):  # pylint: disable=no-self-use
        """
        returns a multiplicative factor to apply to the fourier coeffs before
        computing the inverse transform (trivial in this case, since there's no
        stochastic distro in this case)
        """
        return 1.

    @property
    def C0(self):
        " prefactor of psd"
        return (self.compute_prefactor()/np.sqrt(np.prod(self.size)))**2

    @property
    def abs_q(self):
        " radial distances in q-space"
        q_norm = np.sqrt((self.q[0]**2).reshape((-1, 1))+self.q[1]**2)
        order = np.argsort(q_norm, axis=None)
        # The first entry (for |q| = 0) is rejected, since it's 0 by construct
        return q_norm.flatten()[order][1:]

    @property
    def lambdas(self):
        " radial wavelengths in grid"
        return 2*np.pi/self.abs_q

    def compute_prefactor(self):
        """
        computes the proportionality factor that determines the root mean
        square height assuming that the largest wave length is the full
        domain. This is described for the square of the factor on p R7
        """
        if self.lambda_min is not None:
            q_max = 2*np.pi/self.lambda_min
        else:
            q_max = np.pi*min(self.resolution[0]/self.size[0],
                              self.resolution[1]/self.size[1])
        area = np.prod(self.size)
        if self.rms_height is not None:
            return 2*self.rms_height/np.sqrt(
                self.q_min**(-2*self.hurst)-q_max**(-2*self.hurst)) * \
                np.sqrt(self.hurst*np.pi*area)
        elif self.rms_slope is not None:
            return 2*self.rms_slope/np.sqrt(
                q_max**(2-2*self.hurst)-self.q_min**(2-2*self.hurst)) * \
                np.sqrt((1-self.hurst)*np.pi*area)
        else:
            self.Error('Neither rms height nor rms slope is defined!')

    def generate_phases(self):
        """
        generates appropriate random phases (φ(-q) = -φ(q))
        """
        rand_phase = np.random.rand(*self.resolution)*2*np.pi
        coeffs = np.exp(1j*rand_phase)
        for pos_it, neg_it in self.get_negative_frequency_iterator():
            if pos_it != (0, 0):
                coeffs[neg_it] = coeffs[pos_it].conj()
        if self.resolution[0] % 2 == 0:
            r_2 = self.resolution[0]//2
            coeffs[r_2, 0] = coeffs[r_2, r_2] = coeffs[0, r_2] = 1
        return coeffs

    def generate_amplitudes(self):
        "compute an amplitude distribution"
        q_2 = self.q[0].reshape(-1, 1)**2 + self.q[1]**2
        q_2[0, 0] = 1  # to avoid div by zeros, needs to be fixed after
        self.coeffs *= (q_2)**(-(1+self.hurst)/2)*self.prefactor
        self.coeffs[0, 0] = 0  # et voilà
        # Fix Shannon limit:
        self.coeffs[q_2 > self.q_max**2] = 0

    def get_surface(self, lambda_max=None, lambda_min=None, roll_off=1):
        """
        Computes and returs a NumpySurface object with the specified
        properties. This follows appendices A and B of Persson et al. (2005)

        Persson et al., On the nature of surface roughness with application to
        contact mechanics, sealing, rubber friction and adhesion, J. Phys.:
        Condens. Matter 17 (2005) R1-R62, http://arxiv.org/abs/cond-mat/0502419

        Keyword Arguments:
        lambda_max -- (default None) specifies a cutoff value for the longest
                      wavelength. By default, this is the domain size in the
                      smallest dimension
        lambda_min -- (default None) specifies a cutoff value for the shortest
                      wavelength. by default this is determined by Shannon's
                      Theorem.
        """
        if lambda_max is None:
            lambda_max = self.lambda_max
        if lambda_min is None:
            lambda_min = self.lambda_min

        active_coeffs = self.coeffs.copy()
        q_square = self.q[0].reshape(-1, 1)**2 + self.q[1]**2
        if lambda_max is not None:
            q2_min = (2*np.pi/lambda_max)**2
            # ampli_max = (self.prefactor*2*np.pi/self.size[0] *
            #             q2_min**((-1-self.hurst)/2))
            ampli_max = (q2_min)**(-(1+self.hurst)/2)*self.prefactor
            sl = q_square < q2_min
            ampli = abs(active_coeffs[sl])
            ampli[0] = 1
            active_coeffs[sl] *= roll_off*ampli_max/ampli
        if lambda_min is not None:
            q2_max = (2*np.pi/lambda_min)**2
            active_coeffs[q_square > q2_max] = 0
        active_coeffs *= self.distribution
        area = np.prod(self.size)
        profile = ifftn(active_coeffs, area).real
        self.active_coeffs = active_coeffs
        return NumpySurface(profile, self.size)


class RandomSurfaceGaussian(RandomSurfaceExact):
    """ Metasurface with Gaussian height distribution"""
    def __init__(self, resolution, size, hurst, rms_height=None,
                 rms_slope=None, seed=None, lambda_min=None, lambda_max=None):
        """
        Generates a surface with an Gaussian amplitude distribution
        Keyword Arguments:
        resolution -- Tuple containing number of points in spatial directions.
                      The length of the tuple determines the spatial dimension
                      of the problem (for the time being, only 1D or square 2D)
        size       -- domain size. For multidimensional problems,
                      a tuple can be provided to specify the lenths per
                      dimension. If the tuple has less entries than dimensions,
                      the last value in repeated.
        hurst      -- Hurst exponent
        rms_height -- root mean square asperity height
        rms_slope  -- root mean square slope of surface
        seed       -- (default hash(None)) for repeatability, the random number
                      generator is seeded previous to outputting the generated
                      surface
        lambda_min -- (default None) min wavelength to consider when scaling
                      power spectral density
        lambda_max -- (default None) max wavelength to consider when scaling
                      power spectral density
        """
        super().__init__(resolution, size, hurst, rms_height=rms_height,
                         rms_slope=rms_slope, seed=seed, lambda_min=lambda_min,
                         lambda_max=lambda_max)

    def amplitude_distribution(self):
        """
        updates the amplitudes to be a Gaussian distribution around B(q) from
        Appendix B.
        """
        distr = stats.norm.rvs(size=self.coeffs.shape)
        for pos_it, neg_it in self.get_negative_frequency_iterator():
            distr[neg_it] = distr[pos_it]
        return distr


class CapillaryWavesExact(object):
    """Frozen capillary waves"""
    Error = Exception

    def __init__(self, resolution, size, mass_density, surface_tension, 
                 bending_stiffness, seed=None):
        """
        Generates a surface with an exact power spectrum (deterministic
        amplitude)
        Keyword Arguments:
        resolution        -- Tuple containing number of points in spatial directions.
                             The length of the tuple determines the spatial dimension
                             of the problem (for the time being, only 1D or square 2D)
        size              -- domain size. For multidimensional problems,
                             a tuple can be provided to specify the length per
                             dimension. If the tuple has less entries than dimensions,
                             the last value in repeated.
        mass_density      -- Mass density
        surface_tension   -- Surface tension
        bending_stiffness -- Bending stiffness
        rms_height        -- root mean square asperity height
        rms_slope         -- root mean square slope of surface
        seed              -- (default hash(None)) for repeatability, the random number
                             generator is seeded previous to outputting the generated
                             surface
        """
        if seed is not None:
            np.random.seed(hash(seed))
        if not hasattr(resolution, "__iter__"):
            resolution = (resolution, )
        if not hasattr(size, "__iter__"):
            size = (size, )

        self.dim = len(resolution)
        if self.dim not in (1, 2):
            raise self.Error(
                ("Dimension of this problem is {}. Only 1 and 2-dimensional "
                 "problems are supported").format(self.dim))
        self.resolution = resolution
        tmpsize = list()
        for i in range(self.dim):
            tmpsize.append(size[min(i, len(size)-1)])
        self.size = tuple(tmpsize)

        self.mass_density = mass_density
        self.surface_tension = surface_tension
        self.bending_stiffness = bending_stiffness

        max_pixelsize = max(
            (siz/res for siz, res in zip(self.size, self.resolution)))

        self.q = compute_wavevectors(  # pylint: disable=invalid-name
            self.resolution, self.size, self.dim)
        self.coeffs = self.generate_phases()
        self.generate_amplitudes()
        self.distribution = self.amplitude_distribution()
        self.active_coeffs = None

    def get_negative_frequency_iterator(self):
        " frequency complement"
        def iterator():  # pylint: disable=missing-docstring
            for i in range(self.resolution[0]):
                for j in range(self.resolution[1]//2+1):
                    yield (i, j), (-i, -j)
        return iterator()

    def amplitude_distribution(self):  # pylint: disable=no-self-use
        """
        returns a multiplicative factor to apply to the fourier coeffs before
        computing the inverse transform (trivial in this case, since there's no
        stochastic distro in this case)
        """
        return 1.

    @property
    def abs_q(self):
        " radial distances in q-space"
        q_norm = np.sqrt((self.q[0]**2).reshape((-1, 1))+self.q[1]**2)
        order = np.argsort(q_norm, axis=None)
        # The first entry (for |q| = 0) is rejected, since it's 0 by construct
        return q_norm.flatten()[order][1:]

    def generate_phases(self):
        """
        generates appropriate random phases (φ(-q) = -φ(q))
        """
        rand_phase = np.random.rand(*self.resolution)*2*np.pi
        coeffs = np.exp(1j*rand_phase)
        for pos_it, neg_it in self.get_negative_frequency_iterator():
            if pos_it != (0, 0):
                coeffs[neg_it] = coeffs[pos_it].conj()
        if self.resolution[0] % 2 == 0:
            r_2 = self.resolution[0]//2
            coeffs[r_2, 0] = coeffs[r_2, r_2] = coeffs[0, r_2] = 1
        return coeffs

    def generate_amplitudes(self):
        "compute an amplitude distribution"
        q_2 = self.q[0].reshape(-1, 1)**2 + self.q[1]**2
        q_2[0, 0] = 1  # to avoid div by zeros, needs to be fixed after
        self.coeffs *= 1/(self.mass_density+self.surface_tension*q_2+
                          self.bending_stiffness*q_2*q_2)
        self.coeffs[0, 0] = 0  # et voilà
        # Fix Shannon limit:
        #self.coeffs[q_2 > self.q_max**2] = 0

    def get_surface(self):
        """
        Computes and returs a NumpySurface object with the specified
        properties. This follows appendices A and B of Persson et al. (2005)

        Persson et al., On the nature of surface roughness with application to
        contact mechanics, sealing, rubber friction and adhesion, J. Phys.:
        Condens. Matter 17 (2005) R1-R62, http://arxiv.org/abs/cond-mat/0502419

        Keyword Arguments:
        lambda_max -- (default None) specifies a cutoff value for the longest
                      wavelength. By default, this is the domain size in the
                      smallest dimension
        lambda_min -- (default None) specifies a cutoff value for the shortest
                      wavelength. by default this is determined by Shannon's
                      Theorem.
        """
        active_coeffs = self.coeffs.copy()
        active_coeffs *= self.distribution
        area = np.prod(self.size)
        profile = ifftn(active_coeffs, area).real
        self.active_coeffs = active_coeffs
        return NumpySurface(profile, self.size)
