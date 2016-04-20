#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   fftext.pyx

@author Till Junge <till.junge@altermail.ch>

@date   08 Apr 2016

@brief  Tried extending for fftw3

@section LICENCE

Copyright (C) 2016 Till Junge

fftext.pyx is free software; you can redistribute it and or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation, either version 3, or (at
your option) any later version.

fftext.pyx is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with GNU Emacs; see the file COPYING. If not, write to the
Free Software Foundation, Inc., 59 Temple Place - Suite 330,
Boston, MA 02111-1307, USA.
"""


import numpy as np
cimport numpy as cnp

cdef extern from "<complex>" namespace "std":
    pass

cdef extern from "fftext_cc.hh":
    double complex * fft_r2c_2D_wrap (double *arr, size_t n_row,
                                      size_t n_col)
    double * fft_c2r_2D_wrap (double complex *arr, size_t n_row,
                              size_t n_col)
    double complex * fft_2D_wrap (double complex *arr, size_t n_row,
                                  size_t n_col)

    double complex * ifft_2D_wrap (double complex *arr, size_t n_row,
                                  size_t n_col)

cdef extern from "fftw3.h":
    void fftw_free(void *)

cdef class _destructor:
    cdef void * _data
    def __dealloc__(self):
        if self._data is not NULL:
            fftw_free(self._data)


cdef void set_base(cnp.ndarray arr, void *carr):
    cdef _destructor d = _destructor()
    d._data = <void*> carr
    cnp.set_array_base(arr, d)

cpdef rfft2(double[:, ::1] in_arr):
    """ wrapped fftw3 r2c """

    cdef int n_row = in_arr.shape[0], n_col = in_arr.shape[1]

    cdef double complex * out_arr = fft_r2c_2D_wrap(
        &in_arr[0, 0], n_row, n_col)

    cdef double complex[:, ::1] mv = <double complex [:n_row, :n_col//2+1]> out_arr

    cdef cnp.ndarray arr = np.asarray(mv)
    set_base(arr, out_arr)
    return arr

cpdef rfftn(cnp.ndarray arr):
    if arr.ndim == 2:
        return rfft2(arr)
    else:
        return np.fft.rfftn(arr)

cpdef irfft2(double complex[:, ::1] in_arr, s=None):
    """ wrapped fftw3 r2c ONLY FOR EVEN SIZES"""

    cdef int n_row = in_arr.shape[0], n_col = 2*(in_arr.shape[1] - 1)
    if s is not None:
        n_row, n_col = s

    cdef double * out_arr = fft_c2r_2D_wrap(
        &in_arr[0, 0], n_row, n_col)

    cdef double[:, ::1] mv = <double [:n_row, :n_col]> out_arr

    cdef cnp.ndarray arr = np.asarray(mv)
    set_base(arr, out_arr)
    return arr

cpdef irfftn(cnp.ndarray arr, s=None):
    if arr.ndim == 2:
        return irfft2(arr, s=s)
    else:
        return np.fft.irfftn(arr, s=s)

cpdef fft2(double complex[:, ::1] in_arr):
    """ wrapped fftw3 dft """

    cdef int n_row = in_arr.shape[0], n_col = in_arr.shape[1]

    cdef double complex * out_arr = fft_2D_wrap(
        &in_arr[0, 0], n_row, n_col)

    cdef double complex[:, ::1] mv = <double complex [:n_row, :n_col]> out_arr

    cdef cnp.ndarray arr = np.asarray(mv)
    set_base(arr, out_arr)
    return arr

cpdef ifft2(double complex[:, ::1] in_arr):
    """ wrapped fftw3 dft """

    cdef int n_row = in_arr.shape[0], n_col = in_arr.shape[1]

    cdef double complex * out_arr = ifft_2D_wrap(
        &in_arr[0, 0], n_row, n_col)

    cdef double complex[:, ::1] mv = <double complex [:n_row, :n_col]> out_arr

    cdef cnp.ndarray arr = np.asarray(mv)
    set_base(arr, out_arr)
    return arr
