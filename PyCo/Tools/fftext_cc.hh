#include <complex>

#ifndef _FFTEXT_H_
#define _FFTEXT_H_

using cmplex = std::complex<double>;
cmplex * fft_r2c_2D_wrap (double *arr, size_t n_row, size_t n_col);
cmplex * fft_2D_wrap (cmplex *arr, size_t n_row, size_t n_col);
cmplex * ifft_2D_wrap (cmplex *arr, size_t n_row, size_t n_col);
double * fft_c2r_2D_wrap (cmplex *arr, size_t n_row, size_t n_col);

#endif /* _FFTEXT_H_ */
