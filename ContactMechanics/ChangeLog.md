Change log for ContactMechanics
===============================

v1.4.0 (11June24)
-----------------

- MAINT: Bumped SurfaceTopography to 1.13.0
- MAINT: Switched from muSpectre to muFFT

v1.3.0 (20Apr24)
----------------

- API: Homogeneized solver naming convention to use hypens rather than
  underscores
- MAINT: Bumped NuMPI to 0.5.0
- MAINT: Changed version discovery to use the `DiscoverVersion` package

v1.2.0 (14Jan24)
----------------

- MAINT: Compatiblity with SurfaceTopography 1.12.x
- MAINT: Update NuMPI-dependency to 0.4.0 for modern build system
- BUG: Fixed `make_contact_system` pipeline function

v1.1.1 (29Jan23)
----------------

- BUG: Fixed discover of version when in a git repository that is not the
  source directory of ContactMechanics

v1.1.0 (31Dec22)
----------------

- API: `hard_wall.py` is now installed as `ce_hard_wall`
- BUILD: Changed build/packaging from setuptools to flit
- MAINT: Compatibility with optimizers of scipy 1.93

v1.0.0 (23Jul22)
----------------

- MAINT: Bumped version to 1.0 (otherwise identical to 0.93.0)

v0.93.0 (24May22)
-----------------

- API: `callback` argument of optimizer was renamed to `results_callback`
- ENH: Improved algorithm for pure plastic calculations (#68)
- ENH: Pipeline function for contact calculations (#44, #65)
- ENH: Report bibliography (#52)
- MAINT: Removed buffer copy (#39)
- MAINT: Removed `disp_scale` (#29)
- MAINT: Renamed `prestol` to `forcetol` force consistent naming
  (addressed aspects of #28)

v0.92.0 (12Apr22)
-----------------

- BUG: hard_wall.py under pressure control was not working anymore 
- BUG: hardness was ignored in hard_wall (closes #59)
- TST: test hard_wall.py with pytest
- ENH: Emulation of scanning probe images
- DOC: repaired example notebooks

v0.91.0 (9Sep21)
----------------

- DOC: example file for the use of NuMPI's CCGs 
- TST: testing the NuMPI CCGs on nonadhesive primal and dual contact problems
- ENH: use NuMPI CCG also on nonperiodic problem
- ENH: parallelize the primal (gap as variable) objective
- BUG: repair 1D periodic substrate
- ENH: minimize proxy for primal (minimize wrt. gap) and dual (minimize wrt. pressure) formulation 
- Drop support for Python 3.5
- API: ConstrainedConjugateGradients, minimize_proxy: 
    changed disp0 to initial_displacements. 
- API: ConstrainedConjugateGradients, NonSmooth minimize_proxy: 
    added possibility to directly give initial_forces

v0.90.1 (23Jul20)
-----------------

- MAINT: Bumped muFFT dependency to v0.10.0
- MAINT: Avoid unnecessary buffer copies in FFT

v0.90.0 (29Jun20)
-----------------

- Refactored PyCo code into three separate Python modules:
  SurfaceTopography, ContactMechanics and Adhesion
- muFFT dependency updated to muFFT-0.9.3
- Moved documentation from README.md to the docs folder 

Change log for PyCo (previous name of the package)
==================================================

v0.57.0 (15May20)
-----------------

- MAINT: Support for new, rewritten muFFT bindings
- ENH: Bicubic interpolation of two-dimensional topography maps
- ENH: Fourier derivative of topography
- BUG: Computation of plastic area is now parallelized (#303)
- BUG: Info dictionary mutable from user 

v0.56.0 (26Feb20)
-----------------

- ENH: Change orientation of some readers such that all topographies
       look like the image in Gwyddion when plotted with
       "pcolormesh(t.heights().T)" (#295)
- BUG: Fixes unknown unit "um" when reading mi file (#296)
- BUG: Fixes missing channel name for mi files (#294)
- ENH: generate self-affine random surfaces by specifying the self-affine prefactor (#261, #278, #279)
- BUG: now fourier synthesis can generate Linescans again (#277, #279)


v0.55.0 (14Feb20)
-----------------

- API: Readers now report channel info in ChannelInfo class,
       fixes inconsistencies in reporting channel information (#190, #192, #236)
- ENH: Readers report format identifier and are self-documented (#229, #238)
- ENH: Readers now support Gwyddion's text export format for English and German locale (#230)
- ENH: DI reader now read acquisition date and stores it in the info dictionary
- BUG: DI reader autodetection did not work (#258)
- BUG: Fixes orientation for DI files (#291)
- DOC: Added notebook showing how 2D topographies can be plotted
- TST: Added demo notebook which shows how to plot 2D topographies
- ENH: adhesive ideal plastic simulations with Softwall system (#260, #283)

v0.54.4 (20Dec19)
-----------------

- BUG: Fixes missing 'nb_grid_pts' key in channels from IBW reader

v0.54.3 (20Dec19)
-----------------

- BUG: Fixes assertion because of wrong number of channel names (#252)

v0.54.2 (13Dec19)
-----------------

- BUG: fix rms_laplacian for periodic topographies (#247)

v0.54.1 (13Dec19)
-----------------

- ENH: higher order derivative for periodic surface (#234,#227)
- ENH: new reader for Igor Binary Wave files (IBW) (#224)
- BUG: opdx reader can now handle binary filestreams (#209)
- BUG: store and restore periodic flag in NonuniformTopography (#240)

v0.54.0 (06Dec19)
-----------------

- MAINT: correct installation problems because Eigen repository has moved
- ENH: anisotropic cubic Green's Functions 
- BUG: NPY reader can now handle filestreams (#209, NUMPI/#24)
- BUG: opdx reader can now handle filestreams (#209)

v0.53.1 (21Nov19)
-----------------
- API: Detrended Topographies with mode "center" keep is_periodic property. Other modes lead to is_periodic=False. 
  See pastewka/TopoBank/#347

v0.53.0 (20Nov19)
-----------------

- API: ability to set periodic property of HeightContainer in reader.topography (#198)
- API: default window for computing PSD is choosen according to topography.is_periodic (#217)
- Feature: interpolate_fourier pipeline function
- API: Default of check_boundaries in FreeSystem is False
- Bug fix: fourier synthesis had a padding line in the generation of topographies with odd number of points (#202)
- Bug fix: `topography.rms_curvature` no returns rms_curvature, previously rms_laplacian (#200)
- gnuplot scripts to plot logger output

v0.52.0 (25Aug19)
-----------------

- API: Return contact map (the 'active set') from constrained conjugate gradient
- Bug fix: `assign_patch_numbers` was broken on some configurations since v0.51.2

v0.51.2 (8Aug19)
----------------

- Bug fix: `assign_patch_numbers` crashed for maps larger that 64k x 64k (#191)

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
