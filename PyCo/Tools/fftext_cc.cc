#include <fftw3.h>
#include <omp.h>
#include <iostream>
#include <cassert>
#include <algorithm>

#include <type_traits>


#include <stdexcept>
#include "fftext_cc.hh"


cmplex * fft_r2c_2D_wrap (double *arr, size_t n_row, size_t n_col) {
  // see html/Multi_002dDimensional-DFTs-of-Real-Data.html#Multi_002dDimensional-DFTs-of-Real-Data
  const size_t out_size {n_row * (n_col/2+1)};

  auto out = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * out_size));

  if (out == nullptr) {
    throw std::runtime_error("Couldn't allocate the fucking array");
  }

  /* for real-to-complex algos, the in array is destroyed even in out-of-place
     mode, therefore, i precopy the input into the output and run it in-place
   */
  decltype(arr) in = reinterpret_cast<double*>(out);

  auto const flags = FFTW_ESTIMATE;
  auto err_code = fftw_init_threads();
  if (err_code == 0) {
    throw std::runtime_error("Couldn't init threads");
  }
  fftw_plan_with_nthreads(omp_get_max_threads());
  auto p = fftw_plan_dft_r2c_2d(n_row, n_col,
                                in,
                                out,
                                flags);
  {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i<n_row; ++i) {
      auto write_head = in + i*2*(n_col/2+1);
      auto read_head = arr + i*n_col;
      std::copy(read_head, read_head + n_col, write_head);
    }
  }

  fftw_execute(p);

  fftw_destroy_plan(p);
  return reinterpret_cast<cmplex*>(out);
}

double * fft_c2r_2D_wrap (cmplex *arr, size_t n_row, size_t n_col) {
  // see html/Multi_002dDimensional-DFTs-of-Real-Data.html#Multi_002dDimensional-DFTs-of-Real-Data
  const size_t out_size {n_row * (n_col/2+1)};

  auto out = static_cast<double*>(fftw_malloc(sizeof(fftw_complex) * out_size));

  if (out == nullptr) {
    throw std::runtime_error("Couldn't allocate the fucking array");
  }

  /* for real-to-complex algos, the in array is destroyed even in out-of-place
     mode, therefore, i precopy the input into the output and run it in-place
   */
  auto in = reinterpret_cast<fftw_complex*>(out);

  auto const flags = FFTW_ESTIMATE;
  auto err_code = fftw_init_threads();
  if (err_code == 0) {
    throw std::runtime_error("Couldn't init threads");
  }
  fftw_plan_with_nthreads(omp_get_max_threads());
  auto p = fftw_plan_dft_c2r_2d(n_row, n_col,
                                in,
                                out,
                                flags);

  std::copy(arr, arr + n_row * (n_col/2+1), reinterpret_cast<cmplex*>(in));
  fftw_execute(p);
  auto arr_size = n_col * n_row;

  // this correct the weird packing and prepares a nice np-friendy contiguous array
  {
    //#pragma omp parallel for schedule(static)

    for (size_t i = 0; i < n_row; ++i) {
      for (size_t j = 0; j < n_col; ++j) {
        out[j+i*n_col] = out[j+i*2*(n_col/2 + 1)] /arr_size ;
      }
    }
  }
  //for (size_t i = 0; i< arr_size; ++i) {
  //  out[i] /= arr_size;
  //}

  fftw_destroy_plan(p);

  return out;
}

template <int sign>
cmplex * any_fft_2D_wrap (cmplex *arr, size_t n_row, size_t n_col) {
  // see html/Multi_002dDimensional-DFTs-of-Real-Data.html#Multi_002dDimensional-DFTs-of-Real-Data
  const size_t out_size {n_row * n_col};

  auto out = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * out_size));

  if (out == nullptr) {
    throw std::runtime_error("Couldn't allocate the fucking array");
  }

  auto const flags = FFTW_ESTIMATE;
  auto err_code = fftw_init_threads();
  if (err_code == 0) {
    throw std::runtime_error("Couldn't init threads");
  }
  fftw_plan_with_nthreads(omp_get_max_threads());
  auto p = fftw_plan_dft_2d(n_row, n_col,
                            reinterpret_cast<fftw_complex*>(arr),
                            out,
                            sign,
                            flags);


  fftw_execute(p);
  if (sign == FFTW_BACKWARD) {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < out_size; ++i) {
      reinterpret_cast<cmplex*>(out)[i] /= out_size;
    }
  }

  fftw_destroy_plan(p);

  return reinterpret_cast<cmplex*>(out);
}

cmplex * fft_2D_wrap(cmplex *arr, size_t n_row, size_t n_col) {
  return any_fft_2D_wrap<FFTW_FORWARD>(arr, n_row, n_col);
}
cmplex * ifft_2D_wrap(cmplex *arr, size_t n_row, size_t n_col) {
  return any_fft_2D_wrap<FFTW_BACKWARD>(arr, n_row, n_col);
}

double * fft_c2r_2D_wrap (fftw_complex *arr, size_t n_row, size_t n_col) {
  return nullptr;
}
