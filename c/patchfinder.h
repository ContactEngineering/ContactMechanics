/*
@file   patchfinder.h

@author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>

@date   10 Apr 2017

@brief  Analysis of contact patch geometries

@section LICENCE

Copyright 2015-2018 Till Junge, Lars Pastewka

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

#ifndef __PATCHFINDER_H
#define __PATCHFINDER_H

#ifdef __cplusplus
extern "C" {
#endif

PyObject *assign_patch_numbers(PyObject *self, PyObject *args);
PyObject *assign_segment_numbers(PyObject *self, PyObject *args);
PyObject *shortest_distance(PyObject *self, PyObject *args);
PyObject *closest_patch_map(PyObject *self, PyObject *args);
PyObject *distance_map(PyObject *self, PyObject *args);
PyObject *correlation_function(PyObject *self, PyObject *args);
PyObject *perimeter_length(PyObject *self, PyObject *args);

#ifdef __cplusplus
}
#endif

#endif
