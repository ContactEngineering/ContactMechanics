#
# Helper script to install parallel version of the NetCDF library from the
# sources. This is necessary because parallel compiles (if existing) are
# broken on most distributions.
#

curl https://parallel-netcdf.github.io/Release/pnetcdf-${PNETCDF_VERSION}.tar.gz | tar -xzC ${BUILDDIR} &&
  cd ${BUILDDIR}/pnetcdf-${PNETCDF_VERSION} &&
  CC=mpicc CXX=mpicxx ./configure --disable-fortran --disable-cxx --enable-shared --prefix=${PREFIX} &&
  make &&
  make install &&
  cd -

curl https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.12/hdf5-${HDF5_VERSION}/src/hdf5-${HDF5_VERSION}.tar.gz | tar -xzC ${BUILDDIR} &&
  cd ${BUILDDIR}/hdf5-${HDF5_VERSION} &&
  CC=mpicc CXX=mpicxx ./configure --enable-parallel --prefix=${PREFIX} &&
  make &&
  make install &&
  cd -

# We need to compile NetCDF ourselves because there is no package that has
# parallel PnetCDF and HDF5 enabled.
curl https://www.unidata.ucar.edu/downloads/netcdf/ftp/netcdf-c-${NETCDF4_VERSION}.tar.gz | tar -xzC ${BUILDDIR} &&
  mkdir ${BUILDDIR}/netcdf-c-build &&
  cd ${BUILDDIR}/netcdf-c-build &&
  cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${PREFIX} -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx -DUSE_PARALLEL=ON -DENABLE_PARALLEL4=ON -DENABLE_PNETCDF=ON ${BUILDDIR}/netcdf-c-${NETCDF4_VERSION} &&
  make &&
  make install &&
  cd -

# Install netcdf4-python and make sure that it is compiled (no-binary),
# otherwise it will not have parallel support.
HDF5_DIR=${PREFIX} CC=mpicc python -m pip install --no-binary netCDF4 netCDF4==${NETCDF4_PYTHON_VERSION}
