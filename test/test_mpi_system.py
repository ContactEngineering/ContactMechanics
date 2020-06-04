#
# Copyright 2019-2020 Lars Pastewka
#           2018, 2020 Antoine Sanner
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
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

"""
represents the UseCase of creating System with MPI parallelization
"""

import pytest

from ContactMechanics.Factory import make_system

import numpy as np

import os

DATADIR = os.path.dirname(os.path.realpath(__file__))


@pytest.fixture
def examplefile(comm):
    fn = DATADIR + "/worflowtest.npy"
    res = (128, 64)
    np.random.seed(1)
    data = np.random.random(res)
    data -= np.mean(data)
    if comm.rank == 0:
        np.save(fn, data)

    comm.barrier()
    return (fn, res, data)


# DATAFILE = DATADIR + "/worflowtest.npy"
# @pytest.fixture
# def data(comm):
#    res = (256,256)#(128, 64)
#    np.random.seed(1)
#    data = np.random.random(res)
#    data -= np.mean(data)
#    if comm.Get_rank() == 0:
#        np.save(DATAFILE, data)
#    comm.barrier() # all processors wait on the file to be created
#    return data

def test_make_system_from_file(examplefile, comm):
    """
    longtermgoal for confortable and secure use
    Returns
    -------

    """
    # TODO: test this on npy and nc file
    # Maybe it will be another Function or class
    fn, res, data = examplefile

    system = make_system(substrate="periodic",
                         surface=fn,
                         communicator=comm,
                         physical_sizes=(20., 30.),
                         young=1)

    print(system.__class__)


def test_make_system_from_file_serial(comm_self):
    """
    same as test_make_system_from_file but with the reader being not MPI
    compatible
    Returns
    -------

    """
    pass


# def test_automake_substrate(comm):
#    surface = make_sphere(2, (4,4), (1., 1.), )

def test_hardwall_as_string(comm, examplefile):
    fn, res, data = examplefile
    make_system(substrate="periodic",
                surface=fn,
                physical_sizes=(1., 1.),
                young=1,
                communicator=comm)
