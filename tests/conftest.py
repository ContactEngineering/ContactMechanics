
collect_ignore_glob = ["*MPI_*.py"]

from runtests.mpi import MPITestFixture

comm = MPITestFixture([1,2,3, 4,10], scope='session')

