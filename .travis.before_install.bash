#!/usr/bin/env bash

sudo apt-get update
sudo apt-get install libfftw3-dev
if [ "$WITH_MPI" == "yes" ]; then
  sudo apt-get install openmpi-bin libopenmpi-dev libfftw3-mpi-dev libpnetcdf-dev libpnetcdf0d
fi
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
bash miniconda.sh -b -p $HOME/miniconda3
export PATH="$HOME/miniconda3/bin:$PATH"
hash -r
conda config --set always_yes yes --set changeps1 no
conda update -q conda
conda info -a
conda create -q -n test-environment python=$TRAVIS_PYTHON_VERSION numpy scipy
source activate test-environment
python -m pip install $(grep numpy requirements.txt)
if [ "$WITH_MPI" == "yes" ]; then
  curl https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-${HDF5_VERSION}/src/hdf5-${HDF5_VERSION}.tar.gz | tar -xzC /tmp \
    && cd /tmp/hdf5-${HDF5_VERSION} \
    && CC=mpicc CXX=mpicxx ./configure --enable-parallel --prefix=$HOME/.local \
    && make \
    && make install

  # We need to compile NetCDF ourselves because there is no package that has
  # parallel PnetCDF and HDF5 enabled.
  curl https://www.unidata.ucar.edu/downloads/netcdf/ftp/netcdf-c-${NETCDF4_VERSION}.tar.gz | tar -xzC /tmp \
    && mkdir /tmp/netcdf-c-build \
    && cd /tmp/netcdf-c-build \
    && cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$HOME/.local -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx -DUSE_PARALLEL=ON -DENABLE_PARALLEL4=ON -DENABLE_PNETCDF=ON /tmp/netcdf-c-${NETCDF4_VERSION} \
    && make \
    && make install

  python -m pip install --no-binary mpi4py

  # Install netcdf4-python and make sure that it is compiled (no-binary),
  # otherwise it will not have parallel support.
  HDF5_DIR=$HOME/.local CC=mpicc python -m pip install --no-binary netCDF4 netCDF4==${NETCDF4_PYTHON_VERSION}
fi
