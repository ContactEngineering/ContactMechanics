#
# Copyright 2019-2020 Lars Pastewka
#           2019-2020 Antoine Sanner
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

import pytest
import os

from runtests.mpi.tester import WorldTooSmall, create_comm

import NuMPI
from NuMPI import MPI


def MyMPITestFixture(commsize, scope='function'):
    """
    Create a test fixture for MPI Communicators of various communicator
    sizes.
    """

    @pytest.fixture(params=commsize, scope=scope)
    def fixture(request):
        MPI.COMM_WORLD.barrier()
        # Return an NuMPI stub communicator if we don't have mpi4py
        if not NuMPI._has_mpi4py:
            return MPI.COMM_SELF

        # Try creating a communicator and fallback to NuMPI stub
        try:
            comm, color = create_comm(request.param)

            if color != 0:
                pytest.skip("Not using communicator {}."
                            .format(request.param))
                return None
            else:
                # Turn a None into a NuMPI stub communicator
                if comm is None:
                    comm = MPI.COMM_SELF
                print('MPI communicator: Rank {} (of {}).'
                      .format(comm.rank, comm.size))
                return comm

        except WorldTooSmall:
            pytest.skip("Not using communicator {}.".format(request.param))
            return None

    return fixture


comm = MyMPITestFixture([1, 4], scope='session')
comm_self = MyMPITestFixture([1], scope='session')

maxcomm = MyMPITestFixture([MPI.COMM_WORLD.Get_size()], scope="session")


@pytest.fixture(scope="session")
def file_format_examples():
    return os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        'file_format_examples')
