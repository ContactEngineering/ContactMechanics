/*
@file   bicubic.h

@author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>

@date   31 Mar 2020

@brief  Bicubic interpolation of two-dimensional maps

@section LICENCE

Copyright 2020 Lars Pastewka

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
*/

#ifndef __BICUBIC_H
#define __BICUBIC_H

#include <Python.h>

/* ----------------------------------------------------------------------
 * inlined method, to do matrix-vector multiplication for
 * matrix and vector; square matrix is required and output vector is
 * overwritten.
 * --------------------------------------------------------------------*/
template<typename T>
void mat_mul_vec(int dim, const T *Mat, const T *Vin, T *Vout)
{
  for (int i = 0; i < dim; i++, Vout++){
    *Vout = 0.0;
    for (int j = 0; j < dim; j++, Mat++) {
      *Vout += (*Mat)*Vin[j];
    }
  }
}

#define NPARA (4*4)   // 4^dim
#define NCORN 4

#define WRAPX(x) { while (x >= nx_) x -= nx_; while (x < 0) x += nx_; }
#define WRAPY(y) { while (y >= ny_) y -= ny_; while (y < 0) y += ny_; }

class Bicubic {
 public:
  Bicubic(int, int, double **, bool, bool);
  ~Bicubic();

  void eval(double, double, double &);
  void eval(double, double, double &, double &, double &);
  void eval(double, double, double &, double &, double &, double &, double &, double &);

 protected:
  /* table dimensions */
  int nx_, ny_;

  /* interpolate derivatives */
  bool interp_;

  /* values */
  double **values_;

  /* spline coefficients */
  double ***coeff_;

  /* spline coefficients if lowmem is true */
  double **coeff_lowmem_;

  /* lhs matrix */
  double A_[NPARA][NPARA];

  double **get_spline_coefficients(int r1box, int r2box) {
    if (coeff_) {
      int ibox = ny_*r1box+r2box;
      return coeff_[ibox];
    }
    else {
      compute_spline_coefficients(r1box, r2box, values_, coeff_lowmem_);
      return coeff_lowmem_;
    }
  }

  void compute_spline_coefficients(int nhbox, int ncbox, double **values, double **coeff) {
    const int ix1[NCORN] = { 0,1,1,0 };
    const int ix2[NCORN] = { 0,0,1,1 };

    /*
     * construct the 16 r.h.s. vectors ( 1 for each box ).
     * loop through boxes.
     */ 

    double B[NPARA];

    for (int irow = 0; irow < NCORN; irow++) {
      int nx1  = ix1[irow]+nhbox;
      int nx2  = ix2[irow]+ncbox;
      /* wrap to box */
      WRAPX(nx1);
      WRAPY(nx2);
      /* values of function and derivatives at corner. */
      B[irow   ] = values[nx1][nx2];
      /* interpolate derivatives */
      if (interp_) {
        int nx1p = nx1+1;
        int nx1m = nx1-1;
        int nx2p = nx2+1;
        int nx2m = nx2-1;
        WRAPX(nx1p);
        WRAPX(nx1m);
        WRAPY(nx2p);
        WRAPY(nx2m);
        B[irow+4 ] = (values[nx1p][nx2]-values[nx1m][nx2])/2;
        B[irow+8 ] = (values[nx1][nx2p]-values[nx1][nx2m])/2;
      }
      else {
        B[irow+4 ] = 0.0;
        B[irow+8 ] = 0.0;
      }
      B[irow+12] = 0.0;
    }

    double tmp[NPARA];
    mat_mul_vec(NPARA, &A_[0][0], B, tmp);

    /*
     * get the coefficient values.
     */

    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
        int irow = 4*i+j;
	
        coeff[i][j] = tmp[irow];
      }
    }
  }
};

}

#endif


typedef struct {
  PyObject_HEAD

  Bicubic *map_;

} bicubic_t;

#endif
