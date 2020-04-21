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

#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL PYCO_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

#include "bicubic.h"

/*
 * values are supposed to be of size [0:nx][0:ny]
 */
Bicubic::Bicubic(int nx, int ny, double **values, bool interp, bool lowmem)
{
  const int ix1[NCORN] = { 0,1,1,0 };
  const int ix2[NCORN] = { 0,0,1,1 };

  /*
   * calculate 2-d cubic parameters within each box.
   *
   * normalised coordinates.
   *  4--<--3
   *  |     ^
   *  v     |
   *  1-->--2
   */

  int irow, icol, nx1, nx2, npow1, npow2, npow1m, npow2m;
  int i, j;

  /* --- */

  interp_ = interp;

  nx_ = nx;
  ny_ = ny;
  int nboxs = nx_*ny_;

  /*
   * if lowmem = true then spline coefficients will be computed each
   * time eval is called
   */

  if (lowmem) {
    coeff_ = NULL;
    memory_->create(coeff_lowmem_, 4, 4, "Bicubic::coeff_lowmem");
    values_ = values;
  }
  else {
    values_ = NULL;
    coeff_lowmem_ = NULL;
    memory_->create(coeff_, nboxs, 4, 4, "Bicubic::coeff");
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
    nx1   = ix1[irow];
    nx2   = ix2[irow];
    /* loop through powers of variables. */
    for (npow1 = 0; npow1 <= 3; npow1++) {
      for (npow2 = 0; npow2 <= 3; npow2++) {
                         npow1m = npow1-1;
        if (npow1m < 0)  npow1m = 0;
                         npow2m = npow2-1;
        if (npow2m < 0)  npow2m=0;

        icol = 4*npow1+npow2;

        /* values of products within cubic and derivatives. */
        A_[irow   ][icol] = 1.0*(      pow(nx1,npow1 )      *pow(nx2,npow2 ) );
        A_[irow+4 ][icol] = 1.0*(npow1*pow(nx1,npow1m)      *pow(nx2,npow2 ) );
        A_[irow+8 ][icol] = 1.0*(      pow(nx1,npow1 )*npow2*pow(nx2,npow2m) );
        A_[irow+12][icol] = 1.0*(npow1*pow(nx1,npow1m)*npow2*pow(nx2,npow2m) );
      }
    }
  }


  /*
   * solve by gauss-jordan elimination with full pivoting.
   */

  GaussJordan(NPARA, A_[0], error_);

  if (coeff_) {

    for (int nhbox = 0; nhbox < nx_; nhbox++) {
      for (int ncbox = 0; ncbox < ny_; ncbox++) {
        int icol = ny_*nhbox+ncbox;
        compute_spline_coefficients(nhbox, ncbox, values, coeff_[icol]);
      }
    }

  }
}


Bicubic::~Bicubic()
{
  if (coeff_)
    memory_->destroy(coeff_);
  if (coeff_lowmem_)
    memory_->destroy(coeff_lowmem_);
}


void Bicubic::compute_spline_coefficients(int nhbox, int ncbox, double **values, double **coeff) {
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


void Bicubic::eval(double x, double y, double &f)
{
  int r1box = (int) floor( x );
  int r2box = (int) floor( y );

  /*
   * find which box we're in and convert to normalised coordinates.
   */
  double x1   = x - r1box;
  double x2   = y - r2box;
  WRAPX(r1box);
  WRAPY(r2box);

  /*
   * get spline coefficients
   */
  double **coeffi = get_spline_coefficients(r1box, r2box);

  /*
   * compute splines
   */
  f    = 0.0;
  for (int i = 3; i >= 0; i--) {
    double sf = 0.0;
    for (int j = 3; j >= 0; j--) {
      sf = sf*x2 + coeffi[i][j];
    }
    f = f*x1 + sf;
  }
}


void Bicubic::eval(double x, double y, double &f, double &dfdx, double &dfdy)
{
  int r1box = (int) floor( x );
  int r2box = (int) floor( y );

  /*
   * find which box we're in and convert to normalised coordinates.
   */
  double dx   = x - r1box;
  double dy   = y - r2box;
  WRAPX(r1box);
  WRAPY(r2box);

  /*
   * get spline coefficients
   */
  double **coeffi = get_spline_coefficients(r1box, r2box);

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
      double      coefij = coeffi[i][j];
                  sf   =   sf*dy +   coefij;
      if (j > 0)  sfdy = sfdy*dy + j*coefij;
    }
                f    = f   *dx +   sf;
    if (i > 0)  dfdx = dfdx*dx + i*sf;
                dfdy = dfdy*dx +   sfdy;
  }
}


void Bicubic::eval(double x, double y, double &f,
		   double &dfdx, double &dfdy,
		   double &d2fdxdx, double &d2fdydy, double &d2fdxdy)

{
  int r1box = (int) floor( x );
  int r2box = (int) floor( y );

  /*
   * find which box we're in and convert to normalised coordinates.
   */
  double dx   = x - r1box;
  double dy   = y - r2box;
  WRAPX(r1box);
  WRAPY(r2box);

  /*
   * get spline coefficients
   */
  double **coeffi = get_spline_coefficients(r1box, r2box);

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
    double s2fdydx = 0.0;
    for (int j = 3; j >= 0; j--) {
      double      coefij  = coeffi[i][j];
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

}


/* Allocate new instance */

static PyObject *
bicubic_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
  bicubic_t *self;

  self = (bicubic_t *)type->tp_alloc(type, 0);

  self->error_ = NULL;
  self->memory_ = NULL;
  self->map_ = NULL;

  return (PyObject *) self;
}


/* Release allocated memory */

static void
bicubic_dealloc(bicubic_t *self)
{
  if (self->map_)
    delete self->map_;
  if (self->memory_)
    delete self->memory_;
  if (self->error_)
    delete self->error_;

  self->ob_type->tp_free((PyObject*) self);
}


/* Initialize instance */

static int
bicubic_init(bicubic_t *self, PyObject *args,
		      PyObject *kwargs)
{
  PyObject *py_map_data_in;

  if (!PyArg_ParseTuple(args, "O|O", &py_map_data_in))
    return -1;

  PyObject *py_map_data;
  npy_intp nx, ny;

  py_map_data = PyArray_FROMANY(py_map_data_in, NPY_DOUBLE, 2, 2, 0);
  if (!py_map_data)
    return -1;
  nx = PyArray_DIM(py_map_data, 0);
  ny = PyArray_DIM(py_map_data, 1);

  self->error_ = new LAMMPS_NS::Error();
  self->memory_ = new LAMMPS_NS::Memory(self->error_);

  double **map_data = new double*[nx];
  for (int i = 0; i < nx; i++) {
    map_data[i] = &((double *) PyArray_DATA(py_map_data))[i*ny];
  }

  self->map_ = new LAMMPS_NS::Bicubic(nx, ny, map_data, true, false,
                                      self->error_, self->memory_);

  delete map_data;
  Py_DECREF(py_map_data);

  return 0;
}


/* Call object */

static PyObject *
bicubic_call(bicubic_t *self, PyObject *args,
		      PyObject *kwargs)
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
      PyErr_SetString(PyExc_TypeError, "Map index needs to have x- and y- "
		      "component only.");
      return NULL;
    }

    npy_intp n = PyArray_DIM(py_r, 0);
    double *r = (double *) PyArray_DATA(py_r);

    PyObject *py_v = PyArray_SimpleNew(1, &n, NPY_DOUBLE);
    double *v = (double *) PyArray_DATA(py_v);

    for (int i = 0; i < n; i++) {
      double dx, dy;
      self->map_->eval(r[2*i], r[2*i+1], v[i], dx, dy);
    }

    Py_DECREF(py_r);

    return py_v;
  }
  else if (( PyFloat_Check(py_x)||PyInt_Check(py_x)||PyLong_Check(py_x) ) &&
	   ( PyFloat_Check(py_y)||PyInt_Check(py_y)||PyLong_Check(py_y) )) {
    /* x and y are specified separately, and are scalars */

    double v, dx, dy;
    self->map_->eval(PyFloat_AsDouble(py_x), PyFloat_AsDouble(py_y),
                     v, dx, dy);
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
      PyErr_SetString(PyExc_TypeError, "x- and y-components need to have "
		      "identical number of dimensions.");
      return NULL;
    }

    /* Check that x and y have the same length in each dimension */
    int ndims = PyArray_NDIM(py_xd);
    npy_intp *dims = PyArray_DIMS(py_xd);
    npy_intp n = 1;
    for (int i = 0; i < ndims; i++) {
      npy_intp d = PyArray_DIM(py_yd, i);

      if (dims[i] != d) {
	PyErr_SetString(PyExc_TypeError, "x- and y-components vectors need to "
			"have the same length.");
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
      self->map_->eval(x[i], y[i], v[i], dx, dy);
    }

    Py_DECREF(py_xd);
    Py_DECREF(py_yd);

    return py_v;
  }
}


/* Class declaration */

PyTypeObject bicubic_type = {
    PyObject_HEAD_INIT(NULL)
    0,                                          /* ob_size */
    "bicubic.Bicubic",                          /* tp_name */
    sizeof(bicubic_t),                          /* tp_basicsize */
    0,                                          /* tp_itemsize */
    (destructor)bicubic_dealloc,                /* tp_dealloc */
    0,                                          /* tp_print */
    0,                                          /* tp_getattr */
    0,                                          /* tp_setattr */
    0,                                          /* tp_compare */
    0,                                          /* tp_repr */
    0,                                          /* tp_as_number */
    0,                                          /* tp_as_sequence */
    0,                                          /* tp_as_mapping */
    0,                                          /* tp_hash */
    (ternaryfunc)bicubic_call,                  /* tp_call */
    0,                                          /* tp_str */
    0,                                          /* tp_getattro */
    0,                                          /* tp_setattro */
    0,                                          /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,   /* tp_flags */
    "Potential objects",                        /* tp_doc */
    0,		                                /* tp_traverse */
    0,		                                /* tp_clear */
    0,		                                /* tp_richcompare */
    0,		                                /* tp_weaklistoffset */
    0,		                                /* tp_iter */
    0,		                                /* tp_iternext */
    0,                                          /* tp_methods */
    0,                                          /* tp_members */
    0,                                          /* tp_getset */
    0,                                          /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    (initproc)bicubic_init,                     /* tp_init */
    0,                                          /* tp_alloc */
    bicubic_new,                                /* tp_new */
};
