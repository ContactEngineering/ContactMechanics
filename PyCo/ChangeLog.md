
Change log for PyCo
===================

v0.51.1 (7Aug19)
----------------

- Bug fix: Setting physical_sizes argument in readers (#188)
- Bug fix: physical_sizes should be None for surfacs without a physical size (#189)
- Bug fix: Running and testing without mpi4py is now possible (#179)
- Bug fix: Multiple calls to `topography` method of readers (#187)
- Method to inspect pipeline (#175)
- CI: All tests (serial and MPI parallel) pass in Travis CI

v0.51.0 (5Aug19)
----------------

- Cleanup of new reader API

v0.50.2 (1Aug19)
----------------

- Bug fix: Missing `channel` argument for `topography` method of `WrappedReader` (#181)
- `WrappedReader` now uses 'Default' as channel name

v0.50.1 (1Aug19)
----------------

- Bug fix: Running without an MPI installation
- Bug fix: Reading DI files with non-topographic data (#338)

v0.50.0 (31Jul19)
-----------------

Overview:

- MPI parallelization of topographies, substrates and interaction.
- Updated reader framework that supports loading files in parallel. This requires to peek at the files (without
  loading them) to understand the number of grid points to decide on a domain decomposition strategy.

Technical:

- Use MPI wrapper provided by NuMPI (https://github.com/IMTEK-Simulation/NuMPI) for serial calculations.
- Switch to parallel L-BFGS of NuMPI. 
- Removed Cython dependencies. (Parallel) FFT is now handled by muFFT (https://gitlab.com/muspectre/muspectre).
- Tests have been partially converted to pytest. Parallel tests are run through run-tests
  (https://github.com/AntoineSIMTEK/runtests).

v0.32.0 (15Jul19)
-----------------

- Autocorrelation and power-spectrum updates. Both now have an option 'algorithm' that let's the user select
  between a (fast) FFT and a (slow) brute-force implementation.
  
v0.31.3 (7Jul19)
----------------

- Removed check for existing forces on boundaries (nonperiodic calculations only).

v0.31.1 (20May19)
-----------------

- Bug fix: Contact calculations now also run with detrended/scaled topographies.
- Updated hard wall command line script to new topography interface.

v0.31.0 (5Mar19)
----------------

- Added height-difference autocorrelation and variable bandwidth analysis for nonuniform
  line scans.
- Added wrapper 'to_nonuniform' function that turns uniform into nonuniform line scans.
- Bug fix: 'center' detrend mode for nonunform line scans now minimizes rms height.

v0.30.0 (15Feb19)
-----------------

Overview:

- Added non-uniform line scans, which can be loaded from text files or constructed from arrays.
- New class structure for topographies and line scans (for easier maintenance).
- Major API changes and several bug fixes (see below).
- Added Hardwall simulation tutorial.
- Added calculation for second derivative and RMS curvature for nonuniform topographies.
- Added coordination counting for contact patches. 
- Simplified computation of perimeter using coordination counting.
- Started Sphinx documentation with notes how to use the package.

API Changes:

- New API for generating topographies and line scans (height containers) from data, 
  please use "from PyCo Topography import Topography, NonlinearLineScan, UniformLineScan" now.
- New API for building pipelines using methods on height containers, e.g. "topography.scale(2).detrend()".
- Uniform topographies and line scans can be periodic.
- Removed unit property from height containers. Units are now stored in the info dictionary,
  which has to be set on generation of the height container.
- All topographies must have a physical_sizes. Readers use the resolution as the default physical_sizes 
  if the files contain no physical_sizes information. 
- Removed 'shape' alias to 'resolution' property for height containers.
- Size + shape are now always tuples, physical_sizes is also always set as tuple.
- Topographies can now be pickled and unpickled.  
- Replaced class 'Sphere' with generator function 'make_sphere'.
- Contact with "FreeFFTElasticHalfSpace": 
  Now an error is raised when points at the outer ring of the surface are interacting. 
  See notebook "examples/Hardwall_Simulation.ipynb".
  
Bug fixes:
   
- periodicity was ignored in calculation of the distance between contact patches in `distance_map`
- computation of energy in fourier space didn't match the computation of energy in real space 
  (however it is not used in actual simulation)   
- Removed keyword "full_output" from shift_and_tilt().
- Text files without spaces at beginning of line can be read now.
- Enable reading topography data from memory buffers and from binary streams.
- Calculation of 2D autocorrelation function was broken, e.g. radial average.
- 1D autocorrelation was broken for nonperiodic calculations.


v0.18.0 (31Oct18)
-----------------

- Refactored "Surface" to "Topography".
- Bug fix: Corrected computation of attractive contact area in Smooth contact system.
- Bug fix: Corrected computation of inflexion point in LJ93 and VW82 smoothed potentials.

v0.17.0 (06Jul18)
-----------------

- Height-difference autocorrelation function.

v0.16.0 (23Oct17)
-----------------

- PyCo now licensed under MIT license.

v0.15.0 (06Sep17)
-----------------

- Implemented substrates of finite thickness.
- Support for additional DI file formats.
- More clever unit conversion in DI files.

v0.14.1 (16Jun17)
-----------------

- macOS compatibility fixes.
- Automatic conversion from hardness value (given in units of pressure)
  into internal units in constrained CG solver.

v0.14.0 (14Mar17)
-----------------

- Added penetration hardness model for simple plastic calculations.

v0.13.1 (07Mar17)
-----------------

- Bug fix: Periodic Green's function offset by one lattice constant.

v0.13.0 (13Jan17)
-----------------

- Added further adhesive reference models (Maugis-Dugdale type models for
  cylinder and wedge).
- Added callback option for Polonsky & Keer optimizer.
- setup.py now has '--openmp' option that triggers compilation of shared-memory
  (OpenMP) parallel code.

v0.12.0 (05Dec16)
-----------------

- Main enhancement: Support for masked_arrays in NumpySurface. This allows to
  have undefined (missing) data points in surfaces. Polonsky & Keer can handle
  this now.
- Polonsky & Keer can now optimize at constant pressure (in addition to
  constant displacement)
- Updated hard wall script to accept command line arguments.
- Moved scripts to new 'commandline' folder.
- Added plotmap.py, tool for plotting surfaces from the command line.
- Added plotpsd.py, tool for plotting the PSD of a surface from the command
  line.

v0.11.0 (21Sep16)
-----------------

- Renamed TiltedSurface to DetrendedSurface.

v0.10.3 (16Sep16)
-----------------

- Added reader for HGT files (topography data from NASA Shuttle Radar Topography
  Mission).
- Bug fix in deprecated 'set_size' that broke hard wall example.

v0.10.2 (29Aug16)
-----------------

- Added reader for MATLAB files.

v0.10.1 (22Aug16)
-----------------

- Added 'center' detrending mode which just subtracts the mean value.
- Added getter and setter for detrend_mode.
- Added function to return string representation of subtracted plane.
- Added area_per_pt property to Surface object.

v0.10.0 (31Jul16)
-----------------

- Exponential adhesion potential from Martin's contact mechanics challenge, to
  be used in combination with hard-wall (bounded L-BFGS). Added tests for this
  potential. Thanks go to Joe Monty for implementing this.
- Surfaces now have a *unit* property, that can be any object but will likely
  be a string in many cases.
- Readers now create NumpySurface with *raw* data and wrap it into a
  ScaledSurface to convert to proper unit.
- Travis-CI integration

v0.9.4 (27Apr16)
----------------

- Greenwood-Tripp reference solution
- Many bug fixes in topography file readers

v0.9.3 (20Mar16)
----------------

- Wyko OPD reader (.opd)
- Digital Instruments Nanoscope reader (.di)
- Igor Binary Wave reader (.ibw)
- Detrending

v0.9.2 (06Mar16)
----------------

- X3P reader (.x3p)
- Automatic file format detection
