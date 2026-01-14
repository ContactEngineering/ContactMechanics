import sys

import muGrid

mpi_required = sys.argv[1] in ["yes", "1"]

print('`muGrid` capabilities:')
print(f'* has_mpi: {muGrid.has_mpi}')
print(f'* has_cuda: {muGrid.has_cuda}')

if mpi_required:
    # Make sure that we have the parallel version running
    assert muGrid.has_mpi, "MPI support required but muGrid.has_mpi is False"
