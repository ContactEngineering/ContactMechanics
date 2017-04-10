#ifndef __PATCHFINDER_H
#define __PATCHFINDER_H

static PyObject *assign_patch_numbers(PyObject *self, PyObject *args);
static PyObject *assign_segment_numbers(PyObject *self, PyObject *args);
static PyObject *shortest_distance(PyObject *self, PyObject *args);
static PyObject *distance_map(PyObject *self, PyObject *args);
static PyObject *correlation_function(PyObject *self, PyObject *args);
static PyObject *perimeter_length(PyObject *self, PyObject *args);

#endif
