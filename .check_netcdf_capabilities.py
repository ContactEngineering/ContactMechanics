import sys

import netCDF4

mpi_required = sys.argv[1] in ["yes", "1"]

print('NetCDF configuration:')
print('CDF5 {}'.format(netCDF4.__has_cdf5_format__))
print('Parallel4: {}'.format(netCDF4.__has_parallel4_support__))
print('PnetCDF: {}'.format(netCDF4.__has_pnetcdf_support__))

# Make sure that we have CDF5 support
assert netCDF4.__has_cdf5_format__

if mpi_required:
    # Make sure that we have the parallel version running
    #assert netCDF4.__has_parallel4_support__
    assert netCDF4.__has_pnetcdf_support__
