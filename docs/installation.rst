Installation
============

You need Python 3 and all dependencies of SurfaceTopography_ to install ContactMechanics.
More details on the installation of dependencies are provided in SurfaceTopography's `installation instructions <https://contactengineering.github.io/SurfaceTopography/installation.html>`.


Direct installation with pip
----------------------------

ContactMechanics can be installed by invoking

.. code-block:: bash

    python3 -m pip  install [--user] git+https://github.com/ComputationalMechanics/ContactMechanics.git


Updating ContactMechanics
--------------------------

If you update ContactMechanics (whether with pip or `git pull` if you cloned the repository),  you may need to
uninstall `NuMPI`, `muSpectre` and or `runtests`, so that the newest version of them will be installed.

.. _SurfaceTopography: https://github.com/ContactEngineering/SurfaceTopography
.. _FFTW3: http://www.fftw.org/
.. _muFFT: https://gitlab.com/muspectre/muspectre.git
.. _nuMPI: https://github.com/IMTEK-Simulation/NuMPI.git
.. _runtests: https://github.com/bccp/runtests
.. _Homebrew: https://brew.sh/
.. _OpenBLAS: https://www.openblas.net/
.. _LAPACK: http://www.netlib.org/lapack/