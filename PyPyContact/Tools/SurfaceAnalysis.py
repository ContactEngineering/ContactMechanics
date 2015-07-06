#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   SurfaceAnalysis.py

@author Till Junge <till.junge@kit.edu>

@date   25 Jun 2015

@brief  Provides a tools for the analysis of surface power spectra

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
import numpy as np
from . import compute_wavevectors

class CharacterisePeriodicSurface(object):
    """
    Simple inverse FFT analysis without window. Do not use for measured surfs
    """
    def __init__(self, surface):
        """
        Keyword Arguments:
        surface -- Instance of PyPyContact.Surface or subclass with specified
                   size
        """
        self.surface = surface
        if self.surface.size is None:
            raise Exception("Surface size has to be known (and specified)!")
        if self.surface.dim != 2:
            raise Exception("Only 2D surfaces, for the time being")
        if self.surface.size[0] != self.surface.size[1]:
            raise Exception("Only square surfaces, for the time being")
        if self.surface.resolution[0] != self.surface.resolution[1]:
            raise Exception("Only square surfaces, for the time being")

        self.C, self.q = self.eval()
        self.size = self.C.size

    def eval(self):
        res, size = self.surface.resolution, self.surface.size
        # equivalent lattice constant**2
        a2 = np.prod(size)/np.prod(res)
        h_a = np.fft.fftn(self.surface.profile())
        C_q = (np.conj(h_a)*h_a).real/np.prod(res)

        q_vecs = compute_wavevectors(res, size, self.surface.dim)
        q_norm = np.sqrt((q_vecs[0]**2).reshape((-1, 1))+q_vecs[1]**2)
        order = np.argsort(q_norm, axis=None)
        # The first entry (for |q| = 0) is rejected, since it's 0 by construction
        return C_q.flatten()[order][1:], q_norm.flatten()[order][1:]

    def get_q_from_lambda(self, lambda_min, lambda_max):
        if lambda_min == 0:
            q_max = float('inf')
        else:
            q_max = 2*np.pi/lambda_min
        q_min = 2*np.pi/lambda_max
        return q_min, q_max


    def estimate_hurst(self, lambda_min = 0, lambda_max = float('inf'),
                       full_output=False):
        q_min, q_max = self.get_q_from_lambda(lambda_min, lambda_max)
        sl = np.logical_and(self.q<q_max, self.q>q_min)
        exponent, offset = np.polyfit(np.log(self.q[sl]),
                                      np.log(self.C[sl]), 1)
        prefactor = np.exp(offset)
        Hurst= -(exponent+2)/2
        if full_output:
            return Hurst, prefactor
        else:
            return Hurst

    def h_rms(self):
        return self.surface.h_rms()

    def grouped_stats(self, nb_groups, percentiles=(5, 95)):
        """
        Make nb_groups groups of the ordered C-q dataset and compute their means
        standard errors for plots with errorbars
        Keyword Arguments:
        nb_groups --
        """
        boundaries = np.logspace(np.log10(self.q[0]),
                                 np.log10(self.q[-1]), nb_groups+1)
        C_g = np.zeros(nb_groups)
        C_g_std = np.zeros((len(percentiles), nb_groups))
        q_g = np.zeros(nb_groups)

        for i in range(nb_groups):
            bottom = boundaries[i]
            top = boundaries[i+1]
            sl = np.logical_and(bottom <= self.q, self.q <= top)
            if sl.sum():
                C_sample = self.C[sl]
                q_sample = self.q[sl]

                C_g[i] = C_sample.mean()
                C_g_std[:, i] = abs(np.percentile(C_sample, percentiles)-C_g[i])
                q_g[i] = q_sample.mean()
            else:
                C_g[i] = float('nan')
                C_g_std[:, i] = float('nan')
                q_g[i] = float('nan')

            bottom = top

        return C_g, C_g_std, q_g
