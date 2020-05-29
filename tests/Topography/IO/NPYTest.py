#
# Copyright 2019-2020 Antoine Sanner
#           2019 Lars Pastewka
# 
# ### MIT license
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

import numpy as np
import os

from NuMPI import MPI

from PyCo.SurfaceTopography.IO.NPY import NPYReader
from PyCo.SurfaceTopography.IO.NPY import save_npy
from PyCo.SurfaceTopography import open_topography

import pytest
from NuMPI import MPI
pytestmark = pytest.mark.skipif(MPI.COMM_WORLD.Get_size() > 1,
        reason="tests only serial funcionalities, please execute with pytest")

def test_save_and_load(comm_self, file_format_examples):
    # sometimes the surface isn't transposed the same way when
    topography = open_topography(
        os.path.join(file_format_examples, 'di4.di'),
        format="di").topography()

    npyfile = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                          "test_save_and_load.npy")
    save_npy(npyfile, topography)

    loaded_topography = NPYReader(npyfile, communicator=comm_self).topography(
        #nb_subdomain_grid_pts=topography.nb_grid_pts,
        #subdomain_locations=(0,0),
        physical_sizes=(1., 1.) )

    np.testing.assert_allclose(loaded_topography.heights(), topography.heights())

    os.remove(npyfile)

@pytest.mark.xfail
def test_save_and_load_np(comm_self, file_format_examples):
    # sometimes the surface isn't transposed the same way when

    topography = open_topography(
        os.path.join(file_format_examples, 'di4.di'),
        format="di").topography()

    npyfile = "test_save_and_load_np.npy"
    np.save(npyfile, topography.heights())

    loaded_topography = NPYReader(npyfile, communicator=comm_self).topography(size=(1., 1.))

    np.testing.assert_allclose(loaded_topography.heights(), topography.heights())
    
    os.remove(npyfile)
