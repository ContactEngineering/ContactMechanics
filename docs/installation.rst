Installation
============

You need Python 3 and FFTW3_ to run ContactMechanics. All Python dependencies can be installed
automatically by invoking

Direct installation with pip
----------------------------

ContactMechanics can be installed by invoking

.. code-block:: bash

    python3 -m pip  install [--user] git+https://github.com/ComputationalMechanics/ContactMechanics.git

The command will install other dependencies including muFFT_, NuMPI_ and
runtests_.

Note: sometimes muFFT_ will not find the FFTW3 installation you expect.
You can specify the directory where you installed FFTW3_
by setting the environment variable `FFTWDIR` (e.g. to `$USER/.local`).

Installation from source directory
----------------------------------

If you cloned the repository. You can install the dependencies with

.. code-block:: bash

    python3 -m pip install -r requirements.txt

in the source directory. ContactMechanics can be installed by invoking

.. code-block:: bash

   python3 -m pip install [--user] .

or

.. code-block:: bash

   python3 setup.py install [--user]

in the source directoy. The command line parameter `--user` is optional and leads to a local installation in the current user's `$HOME/.local` directory.

Installation problems with LAPACK and OpenBLAS
-----------------------------------------------

`bicubic.cpp` is linked with `lapack`, that is already available as a dependency of `numpy`.
If during build, `setup.py` fails to link to one of the lapack implementations
provided by `numpy`, as often experienced on macOS, try providing following environment variables:

.. code-block:: bash

    export LDFLAGS="-L/usr/local/opt/openblas/lib $LDFLAGS"
    export CPPFLAGS="-I/usr/local/opt/openblas/include $CPPFLAGS"
    export PKG_CONFIG_PATH="/usr/local/opt/openblas/lib/pkgconfig:$PKG_CONFIG_PATH"

    export LDFLAGS="-L/usr/local/opt/lapack/lib $LDFLAGS"
    export CPPFLAGS="-I/usr/local/opt/lapack/include $CPPFLAGS"
    export PKG_CONFIG_PATH="/usr/local/opt/lapack/lib/pkgconfig:$PKG_CONFIG_PATH"

where the paths have probably to be adapted to your particular installation method.
Here OpenBLAS_ and LAPACK_ was installed via Homebrew_.

Updating ContactMechanics
--------------------------

If you update ContactMechanics (whether with pip or `git pull` if you cloned the repository),  you may need to
uninstall `NuMPI`, `muSpectre` and or `runtests`, so that the newest version of them will be installed.

.. _FFTW3: http://www.fftw.org/
.. _muFFT: https://gitlab.com/muspectre/muspectre.git
.. _nuMPI: https://github.com/IMTEK-Simulation/NuMPI.git
.. _runtests: https://github.com/bccp/runtests
.. _Homebrew: https://brew.sh/
.. _OpenBLAS: https://www.openblas.net/
.. _LAPACK: http://www.netlib.org/lapack/