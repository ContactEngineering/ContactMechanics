/*
@file   autocorrelation.cpp

@author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>

@date   11 Jul 2019

@brief  Height-difference autocorrelation of nonuniform line scans

@section LICENCE

Copyright 2019 Lars Pastewka

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

#include <algorithm>
#include <cmath>

#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL PYCO_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

#include "autocorrelation.h"

#define FINALIZE_AND_RETURN \
  Py_XDECREF(py_acf); \
  Py_XDECREF(py_double_x); \
  Py_XDECREF(py_double_h); \
  Py_XDECREF(py_double_distances); \
  return py_ret;

PyObject *
nonuniform_autocorrelation_1D(PyObject *self, PyObject *args)
{
  PyObject *py_x = NULL, *py_double_x = NULL;
  PyObject *py_h = NULL, *py_double_h = NULL;
  PyObject *py_distances = NULL;
  PyObject *py_acf = NULL;
  PyObject *py_ret = NULL;
  PyObject *py_double_distances = NULL;
  double physical_size;

  py_distances = NULL;
  if (!PyArg_ParseTuple(args, "OOd|O", &py_x, &py_h, &physical_size, &py_distances)) {
    FINALIZE_AND_RETURN;
  }

  py_double_x = (PyObject*) PyArray_FROMANY((PyObject *) py_x, NPY_DOUBLE, 1, 1, NPY_C_CONTIGUOUS);
  if (!py_double_x) {
    FINALIZE_AND_RETURN;
  }

  py_double_h = (PyObject*) PyArray_FROMANY((PyObject *) py_h, NPY_DOUBLE, 1, 1, NPY_C_CONTIGUOUS);
  if (!py_double_h) {
    FINALIZE_AND_RETURN;
  }

  npy_intp nb_grid_pts = PyArray_DIM(py_double_x, 0);
  if (PyArray_DIM(py_double_h, 0) != nb_grid_pts) {
    PyErr_SetString(PyExc_TypeError, "x- and y-arrays must contain identical number of data points.");
  }

  double *x = (double *) PyArray_DATA(py_double_x);
  double *h = (double *) PyArray_DATA(py_double_h);

  double *distances;
  if (py_distances && py_distances != Py_None) {
    py_double_distances = (PyObject*) PyArray_FROMANY((PyObject *) py_distances, NPY_DOUBLE, 1, 1, NPY_C_CONTIGUOUS);
    if (!py_double_distances) {
      FINALIZE_AND_RETURN;
    }
    distances = (double *) PyArray_DATA(py_double_distances);
  }
  else {
    py_double_distances = PyArray_EMPTY(1, &nb_grid_pts, NPY_DOUBLE, 0);
    if (!py_double_distances) {
      FINALIZE_AND_RETURN;
    }
    distances = (double *) PyArray_DATA(py_double_distances);
    /* Create distance array */
    for (int i = 0; i < nb_grid_pts; ++i)  distances[i] = i*physical_size/nb_grid_pts;
  }

  npy_intp nb_distance_pts = PyArray_DIM(py_double_distances, 0);

  py_acf = PyArray_ZEROS(1, &nb_distance_pts, NPY_DOUBLE, 0);
  double *acf = (double *) PyArray_DATA(py_acf);

  for (int i = 0; i < nb_grid_pts-1; ++i) {
    double x1 = x[i];
    double h1 = h[i];
    double s1 = (h[i+1] - h1)/(x[i+1] - x1);
    for (int j = 0; j < nb_grid_pts-1; ++j) {
      /* Determine lower and upper distance between segment i, i+1 and segment j, j+1 */
      double x2 = x[j];
      double h2 = h[j];
      double s2 = (h[j+1] - h[j])/(x[j+1] - x[j]);
      for (int k = 0; k < nb_distance_pts; ++k) {
        double b1 = std::max(x1, x2 - distances[k]);
        double b2 = std::min(x[i + 1], x[j + 1] - distances[k]);
        double b = (b1 + b2) / 2;
        double db = (b2 - b1) / 2;
        if (db > 0) {
          /* f1[x_] := (h1 + s1*(x - x1))
             f2[x_] := (h2 + s2*(x - x2))
             FullSimplify[Integrate[f1[x]*f2[x + d], {x, b - db, b + db}]]
               = 2 * f1[b] * f2[b + d] * db + 2 * s1 * s2 * db ** 3 / 3 */
          double z = h2 - s2 * x2 + (b + distances[k]) * s2 - h1 + s1 * x1 - b * s1;
          double ds = s1 - s2;
          acf[k] += (db * (3 * z * z + ds * ds * db * db)) / 3;
        }
      }
    }
  }

  for (int k = 0; k < nb_distance_pts; ++k)  acf[k] /= (physical_size - distances[k]);

  py_ret = Py_BuildValue("OO", py_double_distances, py_acf);

  FINALIZE_AND_RETURN;
}
