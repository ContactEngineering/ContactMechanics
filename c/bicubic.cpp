/*
@file   bicubic.cpp

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

#include <cmath>
#include <stdexcept>
#include <iostream>

#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL PYCO_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

#include "bicubic.h"

#define DGESV dgesv_

/*
 * signature of dgesv. This should be present once numpy is loaded.
 */
extern "C" void
DGESV(int const* n, int const* nrhs, double* A, int const* lda, int* ipiv, double* B, int const* ldb, int* info);

/*
 * invert a square matrix
 */
int invert_matrix(int n, double *mat)
{
  int n_sq = n*n;

  int ipiv[n];
  int info = 0;

  double tmp1[n_sq], tmp2[n_sq];

  memcpy(tmp1, mat, n_sq*sizeof(double));
  memset(tmp2, 0, n_sq*sizeof(double));
  /* initialize identity matrix */
  for (int i = 0; i < n; i++) {
    tmp2[_row_major(i, i, n, n)] = 1.0;
  }

  DGESV(&n, &n, tmp1, &n, ipiv, tmp2, &n, &info);

  if (info)
    return info;

  memcpy(mat, tmp2, n_sq*sizeof(double));
  return info; /* should be zero */
}

/*
 * values are supposed to be of size [0:nx][0:ny] and stored in row-major order
 */
Bicubic::Bicubic(int n1, int n2, double *values, bool interp, bool lowmem)
  : n1_{n1}, n2_{n2}, coeff_{}, coeff_lowmem_{}
{
  const int box1[NCORN] = { 0,1,1,0 };
  const int box2[NCORN] = { 0,0,1,1 };

  /*
   * calculate 2-d cubic parameters within each box.
   *
   * normalised coordinates.
   *  4--<--3
   *  |     ^
   *  v     |
   *  1-->--2
   */

  int irow, icol, ci1, ci2, npow1, npow2, npow1m, npow2m;

  /* --- */

  this->interp_ = interp;

  /*
   * if lowmem = true then spline coefficients will be computed each
   * time eval is called
   */

  if (lowmem) {
    this->coeff_lowmem_.resize(4*4);
    this->values_ = values;
  }
  else {
    this->values_ = NULL;
    this->coeff_.resize(this->n1_*this->n2_*4*4);
  }

  /*
   * for each box, create and solve the matrix equatoion.
   *    / values of  \     /              \     / function and \
   *  a |  products  | * x | coefficients | = b |  derivative  |
   *    \within cubic/     \ of 2d cubic  /     \    values    /
   */

  /*
   * construct the matrix.
   * this is the same for all boxes as coordinates are normalised.
   * loop through corners.
   */

  for (irow = 0; irow < NCORN; irow++) {
    ci1 = box1[irow];
    ci2 = box2[irow];
    /* loop through powers of variables. */
    for (npow1 = 0; npow1 <= 3; npow1++) {
      for (npow2 = 0; npow2 <= 3; npow2++) {
                         npow1m = npow1-1;
        if (npow1m < 0)  npow1m = 0;
                         npow2m = npow2-1;
        if (npow2m < 0)  npow2m=0;

        icol = _row_major(npow1, npow2, 4, 4);

        /* values of products within cubic and derivatives. */
        A_[irow   ][icol] = 1.0*(      pow(ci1,npow1 )      *pow(ci2,npow2 ) );
        A_[irow+4 ][icol] = 1.0*(npow1*pow(ci1,npow1m)      *pow(ci2,npow2 ) );
        A_[irow+8 ][icol] = 1.0*(      pow(ci1,npow1 )*npow2*pow(ci2,npow2m) );
        A_[irow+12][icol] = 1.0*(npow1*pow(ci1,npow1m)*npow2*pow(ci2,npow2m) );
      }
    }
  }

  /*
   * invert A matrix.
   */

  if (invert_matrix(NPARA, this->A_[0])) {
    throw std::runtime_error("Could not compute spline coefficients.");
  }

  /*
   * if low mem is not requested, we compute all spline coefficients here and store them.
   */

  if (this->coeff_.size()) {
    for (int i1 = 0; i1 < n1_; i1++) {
      for (int i2 = 0; i2 < n2_; i2++) {
        compute_spline_coefficients(i1, i2, values, &this->coeff_[_row_major(i1, i2, this->n1_, this->n2_)]);
      }
    }
  }
}


Bicubic::~Bicubic()
{
}


void
Bicubic::compute_spline_coefficients(int i1, int i2, double *values, double *coeff) {
  const int box1[NCORN] = { 0,1,1,0 };
  const int box2[NCORN] = { 0,0,1,1 };

  /*
   * construct the 16 r.h.s. vectors ( 1 for each box ).
   * loop through boxes.
   */

  double B[NPARA];

  for (int irow = 0; irow < NCORN; irow++) {
    int ci1  = box1[irow]+i1;
    int ci2  = box2[irow]+i2;
    /* wrap to box */
    _wrap(ci1, this->n1_);
    _wrap(ci2, this->n2_);
    /* values of function and derivatives at corner. */
    B[irow   ] = values[_row_major(ci1, ci2, this->n1_, this->n2_)];
    /* interpolate derivatives */
    if (interp_) {
      int ci1p = ci1+1;
      int ci1m = ci1-1;
      int ci2p = ci2+1;
      int ci2m = ci2-1;
      _wrap(ci1p, this->n1_);
      _wrap(ci1m, this->n1_);
      _wrap(ci2p, this->n2_);
      _wrap(ci2m, this->n2_);
      B[irow+4 ] = (
        values[_row_major(ci1p, ci2, this->n1_, this->n2_)] -
        values[_row_major(ci1m, ci2, this->n1_, this->n2_)]
        )/2;
      B[irow+8 ] = (
        values[_row_major(ci1, ci2p, this->n1_, this->n2_)] -
        values[_row_major(ci1, ci2m, this->n1_, this->n2_)]
        )/2;
    }
    else {
      B[irow+4 ] = 0.0;
      B[irow+8 ] = 0.0;
    }
    B[irow+12] = 0.0;
  }

  mat_mul_vec(NPARA, &A_[0][0], B, coeff);
}


void
Bicubic::eval(double x, double y, double &f)
{
  int xbox = static_cast<int>(floor(x));
  int ybox = static_cast<int>(floor(y));

  /*
   * find which box we're in and convert to normalised coordinates.
   */
  double dx = x - xbox;
  double dy = y - ybox;
  _wrap(xbox, this->n1_);
  _wrap(ybox, this->n2_);

  /*
   * get spline coefficients
   */
  const double *coeffi = get_spline_coefficients(xbox, ybox);

  /*
   * compute splines
   */
  f = 0.0;
  for (int i = 3; i >= 0; i--) {
    double sf = 0.0;
    for (int j = 3; j >= 0; j--) {
      sf = sf*dy + coeffi[_row_major(i, j, 4, 4)];
    }
    f = f*dx + sf;
  }
}


void
Bicubic::eval(double x, double y, double &f, double &dfdx, double &dfdy)
{
  int xbox = static_cast<int>(floor(x));
  int ybox = static_cast<int>(floor(y));

  /*
   * find which box we're in and convert to normalised coordinates.
   */
  double dx = x - xbox;
  double dy = y - ybox;
  _wrap(xbox, this->n1_);
  _wrap(ybox, this->n2_);

  /*
   * get spline coefficients
   */
  const double *coeffi = get_spline_coefficients(xbox, ybox);

  /*
   * compute splines
   */
  f    = 0.0;
  dfdx = 0.0;
  dfdy = 0.0;
  for (int i = 3; i >= 0; i--) {
    double sf   = 0.0;
    double sfdy = 0.0;
    for (int j = 3; j >= 0; j--) {
      double      coefij = coeffi[_row_major(i, j, 4, 4)];
                  sf   =   sf*dy +   coefij;
      if (j > 0)  sfdy = sfdy*dy + j*coefij;
    }
                f    = f   *dx +   sf;
    if (i > 0)  dfdx = dfdx*dx + i*sf;
                dfdy = dfdy*dx +   sfdy;
  }
}


void
Bicubic::eval(double x, double y, double &f,
		      double &dfdx, double &dfdy,
		      double &d2fdxdx, double &d2fdydy, double &d2fdxdy)

{
  int xbox = static_cast<int>(floor(x));
  int ybox = static_cast<int>(floor(y));

  /*
   * find which box we're in and convert to normalised coordinates.
   */
  double dx = x - xbox;
  double dy = y - ybox;
  _wrap(xbox, this->n1_);
  _wrap(ybox, this->n2_);

  /*
   * get spline coefficients
   */
  const double *coeffi = get_spline_coefficients(xbox, ybox);

  /*
   * compute splines
   */
  f       = 0.0;
  dfdx    = 0.0;
  dfdy    = 0.0;
  d2fdxdx = 0.0;
  d2fdydy = 0.0;
  d2fdxdy = 0.0;
  for (int i = 3; i >= 0; i--) {
    double sf      = 0.0;
    double sfdy    = 0.0;
    double s2fdydy = 0.0;
    for (int j = 3; j >= 0; j--) {
      double      coefij  = coeffi[_row_major(i, j, 4, 4)];
                  sf      =   sf   *dy +         coefij;
      if (j > 0)  sfdy    = sfdy   *dy + j*      coefij;
      if (j > 1)  s2fdydy = s2fdydy*dy + j*(j-1)*coefij;
    }
                  f       = f      *dx +         sf;
    if (i > 0)    dfdx    = dfdx   *dx + i*      sf;
    if (i > 1)    d2fdxdx = d2fdxdx*dx + i*(i-1)*sf;
                  dfdy    = dfdy   *dx +         sfdy;
    if (i > 0)    d2fdxdy = d2fdxdy*dx + i*      sfdy;
  }
}


/* Allocate new instance */

static PyObject *
bicubic_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
  bicubic_t *self;

  self = (bicubic_t *)type->tp_alloc(type, 0);

  self->map_ = NULL;

  return (PyObject *) self;
}


/* Release allocated memory */

static void
bicubic_dealloc(bicubic_t *self)
{
  if (self->map_)
    delete self->map_;

  Py_TYPE(self)->tp_free((PyObject*) self);
}


/* Initialize instance */

static int
bicubic_init(bicubic_t *self, PyObject *args, PyObject *kwargs)
{
  PyObject *py_map_data_in;

  if (!PyArg_ParseTuple(args, "O|O", &py_map_data_in))
    return -1;

  PyObject *py_map_data;
  npy_intp nx, ny;

  py_map_data = PyArray_FROMANY(py_map_data_in, NPY_DOUBLE, 2, 2, NPY_ARRAY_C_CONTIGUOUS);
  if (!py_map_data)
    return -1;
  nx = PyArray_DIM(py_map_data, 0);
  ny = PyArray_DIM(py_map_data, 1);

  self->map_ = new Bicubic(nx, ny, static_cast<double*>(PyArray_DATA(py_map_data)), true, false);

  Py_DECREF(py_map_data);

  return 0;
}


/* Call object */

static PyObject *
bicubic_call(bicubic_t *self, PyObject *args, PyObject *kwargs)
{
  PyObject *py_x, *py_y;

  /* We support passing coordinates (x, y), numpy arrays (x, y)
     and numpy arrays r */

  py_x = NULL;
  py_y = NULL;
  if (!PyArg_ParseTuple(args, "O|O", &py_x, &py_y))
    return NULL;

  if (!py_y) {
    /* This should a single numpy array r */

    PyObject *py_r;
    py_r = PyArray_FROMANY(py_x, NPY_DOUBLE, 2, 2, 0);
    if (!py_r)
      return NULL;

    if (PyArray_DIM(py_r, 1) != 2) {
      PyErr_SetString(PyExc_TypeError, "Map index needs to have x- and y-component only.");
      return NULL;
    }

    npy_intp n = PyArray_DIM(py_r, 0);
    double *r = (double *) PyArray_DATA(py_r);

    PyObject *py_v = PyArray_SimpleNew(1, &n, NPY_DOUBLE);
    double *v = (double *) PyArray_DATA(py_v);

    for (int i = 0; i < n; i++) {
      self->map_->eval(r[2*i], r[2*i+1], v[i]);
    }

    Py_DECREF(py_r);

    return py_v;
  }
  else if ((PyFloat_Check(py_x) || PyLong_Check(py_x)) && (PyFloat_Check(py_y) || PyLong_Check(py_y))) {
    /* x and y are specified separately, and are scalars */

    double v, dx, dy;
    self->map_->eval(PyFloat_AsDouble(py_x), PyFloat_AsDouble(py_y), v, dx, dy);
    return PyFloat_FromDouble(v);
  }
  else {
    /* x and y are specified separately */
    PyObject *py_xd, *py_yd;
    py_xd = PyArray_FROMANY(py_x, NPY_DOUBLE, 1, 3, 0);
    if (!py_xd)
      return NULL;
    py_yd = PyArray_FROMANY(py_y, NPY_DOUBLE, 1, 3, 0);
    if (!py_yd)
      return NULL;

    /* Check that x and y have the same number of dimensions */
    if (PyArray_NDIM(py_xd) != PyArray_NDIM(py_yd)) {
      PyErr_SetString(PyExc_TypeError, "x- and y-components need to have identical number of dimensions.");
      return NULL;
    }

    /* Check that x and y have the same length in each dimension */
    int ndims = PyArray_NDIM(py_xd);
    npy_intp *dims = PyArray_DIMS(py_xd);
    npy_intp n = 1;
    for (int i = 0; i < ndims; i++) {
      npy_intp d = PyArray_DIM(py_yd, i);

      if (dims[i] != d) {
	    PyErr_SetString(PyExc_TypeError, "x- and y-components vectors need to have the same length.");
	    return NULL;
      }

      n *= d;
    }

    double *x = (double *) PyArray_DATA(py_xd);
    double *y = (double *) PyArray_DATA(py_yd);

    PyObject *py_v = PyArray_SimpleNew(ndims, dims, NPY_DOUBLE);
    double *v = (double *) PyArray_DATA(py_v);

    for (int i = 0; i < n; i++) {
      double dx, dy;
      self->map_->eval(x[i], y[i], v[i]);
    }

    Py_DECREF(py_xd);
    Py_DECREF(py_yd);

    return py_v;
  }
}


/* Class declaration */

PyTypeObject bicubic_type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "bicubic.Bicubic",
    .tp_basicsize = sizeof(bicubic_t),
    .tp_dealloc = (destructor)bicubic_dealloc,
    .tp_call = (ternaryfunc)bicubic_call,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc = "Bicubic interpolation of two-dimensional maps",
    .tp_init = (initproc)bicubic_init,
    .tp_new = bicubic_new,
};
