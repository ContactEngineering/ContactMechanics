"""
This is a demo for checking how run-tests behaves when not all ranks fail.
The test will not deadlock, but the ranks that succeeded during one test will
fail on a subsequent test, and this may make the test report harder to
interprete.

Parallelising assertions is hence not mandatory but recommended.
"""

from NuMPI.Tools import Reduction
import numpy as np


def test_parallel_failure(comm):
    # will this test fail properly or just deadlock ?
    assert comm.rank < 1
    # NICE it doesn't deadlock !


#        z is not a typo, it is for making the test execute after the upper one
def test_z_continue_another_test(comm):
    # however rank 0 only fails on this test.
    pnp = Reduction(comm)
    pnp.sum(np.array([3, 4]))
    assert True


def test_a_clean_way(comm):
    pnp = Reduction(comm)
    pnp.all(True)
