#!/usr/bin/env bash

sudo apt-get update
sudo apt-get install libfftw3-dev libopenblas-dev
if [ "$WITH_MPI" == "yes" ]; then
  sudo apt-get install openmpi-bin libopenmpi-dev libfftw3-mpi-dev
fi
#wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
#bash miniconda.sh -b -p $HOME/miniconda3
#export PATH="$HOME/miniconda3/bin:$PATH"
#hash -r
#conda config --set always_yes yes --set changeps1 no
#conda update -q conda
#conda info -a
#conda create -q -n test-environment python=$TRAVIS_PYTHON_VERSION numpy scipy
#source activate test-environment
python -m pip install $(grep numpy requirements.txt)
if [ "$WITH_MPI" == "yes" ]; then
  python -m pip install --no-binary mpi4py mpi4py==${MPI4PY_VERSION}
  BUILDDIR=/tmp PREFIX=$HOME/.local source .install_parallel_netcdf.sh
fi
